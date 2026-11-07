// Image input for the vision tower: decoding, the Qwen2-VL resize rule, normalization and patch layout,
// plus the per-patch tables the encoder needs (learned position embeddings and 2D rotary angles).
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "model.h"

struct Rgb8 {
    int w = 0, h = 0;
    std::vector<uint8_t> px;  // row-major RGB
};

// Decodes PNG / JPEG / BMP / GIF / TIFF (and WebP / HEIC with the Windows codecs), applying the EXIF
// orientation. Throws on failure.
Rgb8 load_image(const std::string& path);
bool is_image_path(const std::string& path);

struct ImageInput {
    std::string path;
    int width = 0, height = 0;   // original size
    int rw = 0, rh = 0;          // resized: multiples of patch * merge
    int gh = 0, gw = 0;          // patch grid
    std::vector<float> patches;  // [gh*gw][patch_dim]; each 2x2 merge block is 4 consecutive patches
    std::vector<float> pos;      // [gh*gw][hidden] learned position embeddings resampled to the grid
    std::vector<float> rope;     // [gh*gw][head_dim / 2][2] (cos, sin) of the 2D rotary embedding
    int lgh() const { return gh / 2; }  // grid of merged tokens seen by the text model
    int lgw() const { return gw / 2; }
    int tokens() const { return lgh() * lgw(); }
};

// Both sides multiples of `factor`, area within [min_px, max_px], aspect ratio kept (Qwen2-VL smart_resize).
void smart_resize(int h, int w, int factor, int64_t min_px, int64_t max_px, int& rh, int& rw);

ImageInput prepare_image(const Rgb8& img, const Config::Vision& vc, const VisionWeights& vw, int64_t max_pixels);
