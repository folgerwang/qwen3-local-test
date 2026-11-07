#include "vision.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <wincodec.h>
#pragma comment(lib, "windowscodecs.lib")
#pragma comment(lib, "ole32.lib")
#endif

bool is_image_path(const std::string& path) {
    const size_t dot = path.find_last_of('.');
    if (dot == std::string::npos) return false;
    std::string ext = path.substr(dot + 1);
    for (char& c : ext) c = (char)std::tolower((unsigned char)c);
    for (const char* e : {"png", "jpg", "jpeg", "bmp", "gif", "tif", "tiff", "webp", "heic", "jfif"})
        if (ext == e) return true;
    return false;
}

#ifdef _WIN32
namespace {
template <class T>
struct Com {  // minimal COM pointer
    T* p = nullptr;
    ~Com() { if (p) p->Release(); }
    T** operator&() { return &p; }
    T* operator->() const { return p; }
};
void check(HRESULT hr, const char* what, const std::string& path) {
    if (FAILED(hr)) {
        char b[16];
        std::snprintf(b, sizeof b, "%08lx", (unsigned long)hr);
        throw std::runtime_error(std::string("cannot read image ") + path + " (" + what + ", 0x" + b + ")");
    }
}
}  // namespace

Rgb8 load_image(const std::string& path) {
    const HRESULT ci = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    (void)ci;  // S_FALSE / RPC_E_CHANGED_MODE: COM is already usable on this thread
    const int wl = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::wstring wpath(wl, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wl);

    Com<IWICImagingFactory> f;
    check(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&f)), "WIC", path);
    Com<IWICBitmapDecoder> dec;
    check(f->CreateDecoderFromFilename(wpath.c_str(), nullptr, GENERIC_READ, WICDecodeMetadataCacheOnDemand, &dec), "decode", path);
    Com<IWICBitmapFrameDecode> frame;
    check(dec->GetFrame(0, &frame), "frame", path);

    // EXIF orientation (photos from phones), applied like PIL.ImageOps.exif_transpose
    WICBitmapTransformOptions tr = WICBitmapTransformRotate0;
    {
        Com<IWICMetadataQueryReader> q;
        if (SUCCEEDED(frame->GetMetadataQueryReader(&q))) {
            PROPVARIANT v;
            PropVariantInit(&v);
            if (SUCCEEDED(q->GetMetadataByName(L"System.Photo.Orientation", &v))) {
                USHORT o = 1;
                if (v.vt == VT_UI2) o = v.uiVal;
                switch (o) {
                    case 2: tr = WICBitmapTransformFlipHorizontal; break;
                    case 3: tr = WICBitmapTransformRotate180; break;
                    case 4: tr = WICBitmapTransformFlipVertical; break;
                    case 5: tr = (WICBitmapTransformOptions)(WICBitmapTransformRotate90 | WICBitmapTransformFlipHorizontal); break;
                    case 6: tr = WICBitmapTransformRotate90; break;
                    case 7: tr = (WICBitmapTransformOptions)(WICBitmapTransformRotate270 | WICBitmapTransformFlipHorizontal); break;
                    case 8: tr = WICBitmapTransformRotate270; break;
                    default: break;
                }
            }
            PropVariantClear(&v);
        }
    }
    Com<IWICBitmapSource> rgb;
    check(WICConvertBitmapSource(GUID_WICPixelFormat24bppRGB, frame.p, &rgb), "convert", path);
    Com<IWICBitmapFlipRotator> rot;
    IWICBitmapSource* src = rgb.p;
    if (tr != WICBitmapTransformRotate0) {
        check(f->CreateBitmapFlipRotator(&rot), "rotate", path);
        check(rot->Initialize(rgb.p, tr), "rotate", path);
        src = rot.p;
    }
    UINT w = 0, h = 0;
    check(src->GetSize(&w, &h), "size", path);
    if (!w || !h || (uint64_t)w * h > (1ull << 28)) throw std::runtime_error("unsupported image size: " + path);
    Rgb8 img;
    img.w = (int)w;
    img.h = (int)h;
    img.px.resize((size_t)w * h * 3);
    check(src->CopyPixels(nullptr, w * 3, (UINT)img.px.size(), img.px.data()), "pixels", path);
    return img;
}
#else
Rgb8 load_image(const std::string& path) {
    throw std::runtime_error("image loading is only implemented on Windows: " + path);
}
#endif

void smart_resize(int h, int w, int factor, int64_t min_px, int64_t max_px, int& rh, int& rw) {
    if ((double)std::max(h, w) / std::min(h, w) > 200) throw std::runtime_error("image aspect ratio above 200");
    auto round_half_even = [](double v) { return std::nearbyint(v); };  // Python round()
    int64_t hb = (int64_t)round_half_even((double)h / factor) * factor, wb = (int64_t)round_half_even((double)w / factor) * factor;
    if (hb * wb > max_px) {
        const double beta = std::sqrt((double)h * w / (double)max_px);
        hb = std::max<int64_t>(factor, (int64_t)std::floor(h / beta / factor) * factor);
        wb = std::max<int64_t>(factor, (int64_t)std::floor(w / beta / factor) * factor);
    } else if (hb * wb < min_px) {
        const double beta = std::sqrt((double)min_px / ((double)h * w));
        hb = (int64_t)std::ceil(h * beta / factor) * factor;
        wb = (int64_t)std::ceil(w * beta / factor) * factor;
    }
    rh = (int)hb;
    rw = (int)wb;
}

namespace {

// One antialiased bicubic pass over 8-bit pixels, as torchvision resize(antialias=True) does on uint8
// (the Pillow algorithm: Keys a = -0.5, support widened by the downscale factor, int16 fixed-point weights,
// rounding to uint8 after each pass).
struct Taps {
    std::vector<int> start, count;
    std::vector<int16_t> w;  // count[i] weights per output at i * ksize
    int ksize = 0, prec = 0;
};

Taps bicubic_taps(int in, int out) {
    auto cubic = [](double x) {
        const double a = -0.5;
        x = std::fabs(x);
        if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
        if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
        return 0.0;
    };
    const double scale = (double)in / out, fscale = std::max(scale, 1.0), support = 2.0 * fscale;
    Taps t;
    t.ksize = (int)std::ceil(support) * 2 + 1;
    t.start.resize(out);
    t.count.resize(out);
    std::vector<double> wd((size_t)out * t.ksize, 0.0);
    double max_w = 0;
    for (int i = 0; i < out; i++) {
        const double center = (i + 0.5) * scale, ss = 1.0 / fscale;
        int xmin = std::max((int)(center - support + 0.5), 0);
        int xmax = std::min((int)(center + support + 0.5), in) - xmin;
        double total = 0;
        double* k = &wd[(size_t)i * t.ksize];
        for (int x = 0; x < xmax; x++) total += (k[x] = cubic((x + xmin - center + 0.5) * ss));
        for (int x = 0; x < xmax; x++) {
            if (total != 0) k[x] /= total;
            max_w = std::max(max_w, k[x]);
        }
        t.start[i] = xmin;
        t.count[i] = xmax;
    }
    // the largest precision (up to 22 bits) whose weights still fit int16 (torch's uint8 AVX resampler)
    t.prec = 0;
    for (; t.prec < 22; t.prec++)
        if ((int)(0.5 + max_w * (1 << (t.prec + 1))) >= (1 << 15)) break;
    t.w.resize(wd.size());
    for (size_t i = 0; i < wd.size(); i++)
        t.w[i] = (int16_t)(wd[i] * (1 << t.prec) + (wd[i] < 0 ? -0.5 : 0.5));
    return t;
}

uint8_t clip8(int64_t v, int prec) {
    v = (v + ((int64_t)1 << (prec - 1))) >> prec;
    return (uint8_t)std::clamp<int64_t>(v, 0, 255);
}

Rgb8 resize_bicubic(const Rgb8& src, int ow, int oh) {
    Rgb8 cur = src;
    if (ow != cur.w) {  // horizontal pass first, like Pillow / torchvision
        const Taps t = bicubic_taps(cur.w, ow);
        Rgb8 o;
        o.w = ow;
        o.h = cur.h;
        o.px.resize((size_t)ow * o.h * 3);
        for (int y = 0; y < o.h; y++)
            for (int x = 0; x < ow; x++)
                for (int c = 0; c < 3; c++) {
                    int64_t acc = 0;
                    const int16_t* k = &t.w[(size_t)x * t.ksize];
                    for (int i = 0; i < t.count[x]; i++) acc += (int64_t)k[i] * cur.px[((size_t)y * cur.w + t.start[x] + i) * 3 + c];
                    o.px[((size_t)y * ow + x) * 3 + c] = clip8(acc, t.prec);
                }
        cur = std::move(o);
    }
    if (oh != cur.h) {
        const Taps t = bicubic_taps(cur.h, oh);
        Rgb8 o;
        o.w = cur.w;
        o.h = oh;
        o.px.resize((size_t)o.w * oh * 3);
        for (int y = 0; y < oh; y++) {
            const int16_t* k = &t.w[(size_t)y * t.ksize];
            for (int x = 0; x < o.w; x++)
                for (int c = 0; c < 3; c++) {
                    int64_t acc = 0;
                    for (int i = 0; i < t.count[y]; i++) acc += (int64_t)k[i] * cur.px[((size_t)(t.start[y] + i) * cur.w + x) * 3 + c];
                    o.px[((size_t)y * o.w + x) * 3 + c] = clip8(acc, t.prec);
                }
        }
        cur = std::move(o);
    }
    return cur;
}

}  // namespace

ImageInput prepare_image(const Rgb8& img, const Config::Vision& vc, const VisionWeights& vw, int64_t max_pixels) {
    const int P = vc.patch, M = vc.merge, T = vc.temporal, VH = vc.hidden, HD = vc.hidden / vc.heads;
    ImageInput in;
    in.width = img.w;
    in.height = img.h;
    smart_resize(img.h, img.w, P * M, vc.min_pixels, max_pixels, in.rh, in.rw);
    const Rgb8 r = resize_bicubic(img, in.rw, in.rh);
    in.gh = in.rh / P;
    in.gw = in.rw / P;
    const int N = in.gh * in.gw, PD = vc.patch_dim(), bw = in.gw / M;

    // patch n in merge-block order -> grid (row, col)
    std::vector<int> row(N), col(N);
    for (int n = 0; n < N; n++) {
        const int ic = n % M, ir = (n / M) % M, bc = (n / (M * M)) % bw, br = n / (M * M * bw);
        row[n] = br * M + ir;
        col[n] = bc * M + ic;
    }

    // pixels: (v / 255 - mean) / std, computed as HF does with the rescale folded in: (v - 255 mean) / (255 std).
    // Each patch is [channel][frame][py][px] with the frame repeated.
    in.patches.resize((size_t)N * PD);
    float mean255[3], std255[3];
    for (int c = 0; c < 3; c++) {
        mean255[c] = vc.mean[c] * 255.0f;
        std255[c] = vc.std[c] * 255.0f;
    }
    for (int n = 0; n < N; n++) {
        float* d = &in.patches[(size_t)n * PD];
        for (int c = 0; c < 3; c++)
            for (int t = 0; t < T; t++)
                for (int py = 0; py < P; py++)
                    for (int px = 0; px < P; px++) {
                        const uint8_t v = r.px[((size_t)(row[n] * P + py) * r.w + col[n] * P + px) * 3 + c];
                        d[((c * T + t) * P + py) * P + px] = ((float)v - mean255[c]) / std255[c];
                    }
    }

    // learned positions: bilinear (align_corners) resampling of the square table to the patch grid
    const int side = (int)std::lround(std::sqrt((double)vc.n_pos));
    auto taps = [&](int idx, int size, int& t0, int& t1, float& w0, float& w1) {
        const float src = (float)idx * (float)(side - 1) / (float)std::max(size - 1, 1);
        const float fl = std::floor(src);
        t0 = std::clamp((int)fl, 0, side - 1);
        t1 = std::clamp((int)fl + 1, 0, side - 1);
        w0 = std::max(0.0f, 1.0f - std::fabs(src - fl));
        w1 = std::max(0.0f, 1.0f - std::fabs(src - fl - 1.0f));
    };
    in.pos.assign((size_t)N * VH, 0.0f);
    for (int n = 0; n < N; n++) {
        int h0, h1, w0i, w1i;
        float hw0, hw1, ww0, ww1;
        taps(row[n], in.gh, h0, h1, hw0, hw1);
        taps(col[n], in.gw, w0i, w1i, ww0, ww1);
        const int idx[4] = {h0 * side + w0i, h0 * side + w1i, h1 * side + w0i, h1 * side + w1i};
        const float wt[4] = {hw0 * ww0, hw0 * ww1, hw1 * ww0, hw1 * ww1};
        float* d = &in.pos[(size_t)n * VH];
        for (int k = 0; k < 4; k++) {
            const uint16_t* e = vw.pos_embed.w + (size_t)idx[k] * VH;
            for (int i = 0; i < VH; i++) d[i] += bf16_to_f32(e[i]) * wt[k];
        }
    }

    // 2D rotary: head_dim/4 frequencies, the first half of the rotated dims follow the row, the second the column
    const int nf = HD / 4;
    std::vector<float> inv(nf);
    for (int i = 0; i < nf; i++) inv[i] = 1.0f / std::pow(vc.rope_theta, (float)(2 * i) / (float)(HD / 2));
    in.rope.resize((size_t)N * (HD / 2) * 2);
    for (int n = 0; n < N; n++)
        for (int j = 0; j < HD / 2; j++) {
            const float a = (j < nf ? (float)row[n] : (float)col[n]) * inv[j % nf];
            in.rope[((size_t)n * (HD / 2) + j) * 2] = std::cos(a);
            in.rope[((size_t)n * (HD / 2) + j) * 2 + 1] = std::sin(a);
        }
    return in;
}
