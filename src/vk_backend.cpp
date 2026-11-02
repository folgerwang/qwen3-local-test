// Vulkan compute backend.
//
// Memory placement
//   * Dense weights (attention, DeltaNet, shared experts, router, lm_head) live in VRAM.
//   * Every routed expert has a copy in pinned host memory that the GPU can read directly over PCIe,
//     laid out exactly like a VRAM cache slot. A VRAM cache of fixed-size slots holds the hot experts.
//   * A per-(layer, expert) location table on the GPU holds the address the shaders read from: the
//     cache slot when cached, the pinned host copy otherwise.
//
// Execution: routing, grouping and expert dispatch happen on the GPU (indirect dispatch), so a forward
// pass is a single submission. Afterwards the CPU reads the selected expert ids, updates LRU state and
// usage counts, and schedules copies of missed experts into cache slots at the start of the next pass.
// Hybrid mode (models whose experts mostly do not fit in VRAM): each layer syncs once after routing; the
// GPU runs the cached experts while the CPU computes the missed ones straight from RAM (no PCIe weight
// traffic), then the GPU combines both. No pinned copy is needed in this mode.
// If pinned memory is unavailable in non-hybrid mode, the backend falls back to a CPU sync per layer
// that uploads misses before the experts run.
//
// Shaders get every buffer as a device address in push constants; there are no descriptor sets.
#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "backend.h"
#include "cpu_kernels.h"
#include "shaders_spv.h"

namespace {

#define VK_CHECK(x)                                                                                     \
    do {                                                                                                \
        VkResult r_ = (x);                                                                              \
        if (r_ != VK_SUCCESS) throw std::runtime_error("Vulkan error " + std::to_string(r_) + ": " #x); \
    } while (0)

constexpr int NMAX = 1024;            // max tokens per forward chunk (big chunks reuse expert weights more)
constexpr int NT = 8;                 // tokens per matvec pass (matches matmul_fp8 / matmul_bf16)
constexpr int GEMM_MIN = 9;           // dense matrices use the tiled GEMM from this many tokens
constexpr int EXPERT_GEMM_MIN = 33;   // grouped experts use the tiled GEMM from this many tokens
constexpr int ATT_PART_MAX = 2048;    // max (tokens x key splits) for split-K attention scratch
constexpr int MAX_UPLOADS = 32;       // cache fills per forward pass (bounds added latency)
constexpr int FILL_MB = 32;           // cache-fill byte budget per pass

constexpr int MAX_UPLOADS_HYBRID = 16; // same in hybrid mode, where a fill is a CPU memcpy + upload
constexpr VkDeviceSize CHUNK = 1ull << 30;
constexpr VkDeviceSize STAGING = 256ull << 20;

enum class Mem { Device, HostUpload, HostRead };

struct Buf {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    uint64_t addr = 0;
    uint8_t* map = nullptr;
};

// A sub-range of a Buf.
struct Ptr {
    Buf* b = nullptr;
    VkDeviceSize off = 0;
    uint64_t addr() const { return b ? b->addr + off : 0; }
};

// ---------------------------------------------------------------------------------------------
// Vulkan device + helpers
// ---------------------------------------------------------------------------------------------
class Gpu {
public:
    Gpu() {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "qwen-local";
        app.apiVersion = VK_API_VERSION_1_3;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
        VK_CHECK(vkCreateInstance(&ici, nullptr, &instance_));

        uint32_t n = 0;
        vkEnumeratePhysicalDevices(instance_, &n, nullptr);
        std::vector<VkPhysicalDevice> devs(n);
        vkEnumeratePhysicalDevices(instance_, &n, devs.data());
        int best = -1;
        for (uint32_t i = 0; i < n; i++) {
            VkPhysicalDeviceProperties p;
            vkGetPhysicalDeviceProperties(devs[i], &p);
            if (p.apiVersion < VK_API_VERSION_1_3) continue;
            if (best < 0 || p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) best = (int)i;
        }
        if (best < 0) throw std::runtime_error("no Vulkan 1.3 device found");
        phys_ = devs[best];
        vkGetPhysicalDeviceProperties(phys_, &props_);
        vkGetPhysicalDeviceMemoryProperties(phys_, &memprops_);

        VkPhysicalDeviceSubgroupSizeControlProperties sscp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &sscp;
        vkGetPhysicalDeviceProperties2(phys_, &p2);
        if (sscp.minSubgroupSize > 32 || sscp.maxSubgroupSize < 32 || !(sscp.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT))
            throw std::runtime_error("device cannot run compute shaders with subgroup size 32");

        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys_, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qf(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(phys_, &qn, qf.data());
        qfam_ = UINT32_MAX;
        for (uint32_t i = 0; i < qn; i++)
            if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfam_ = i; break; }
        if (qfam_ == UINT32_MAX) throw std::runtime_error("no compute queue");

        uint32_t ne = 0;
        vkEnumerateDeviceExtensionProperties(phys_, nullptr, &ne, nullptr);
        std::vector<VkExtensionProperties> exts(ne);
        vkEnumerateDeviceExtensionProperties(phys_, nullptr, &ne, exts.data());
        std::vector<const char*> enable;
        bool has_coop_ext = false;
        for (auto& e : exts) {
            if (!std::strcmp(e.extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) { has_budget_ = true; enable.push_back(e.extensionName); }
            if (!std::strcmp(e.extensionName, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)) has_coop_ext = true;
        }

        // Hardware feature detection: tensor cores via cooperative matrix (f16 x f16 -> f32, 16x16x16, subgroup)
        VkPhysicalDeviceCooperativeMatrixFeaturesKHR cmf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
        VkPhysicalDeviceVulkan12Features q12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        q12.pNext = has_coop_ext ? &cmf : nullptr;
        VkPhysicalDeviceFeatures2 qf2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        qf2.pNext = &q12;
        vkGetPhysicalDeviceFeatures2(phys_, &qf2);
        if (has_coop_ext && cmf.cooperativeMatrix && q12.shaderFloat16 && !std::getenv("QW_NO_COOP")) {
            auto getp = (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)vkGetInstanceProcAddr(
                instance_, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
            uint32_t nc = 0;
            if (getp) getp(phys_, &nc, nullptr);
            std::vector<VkCooperativeMatrixPropertiesKHR> cp(nc, {VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR});
            if (nc) getp(phys_, &nc, cp.data());
            for (auto& m : cp) {
                if (m.MSize == 16 && m.NSize == 16 && m.KSize == 16 && m.AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                    m.BType == VK_COMPONENT_TYPE_FLOAT16_KHR && m.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                    m.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR && m.scope == VK_SCOPE_SUBGROUP_KHR)
                    coop_ = true;
                if ((int)m.AType == 1000491002) coop_fp8_ = true;  // VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT
            }
        }
        if (coop_) enable.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
        // a transfer-only queue family = dedicated copy engine for asynchronous expert uploads
        for (uint32_t i = 0; i < qn; i++)
            if ((qf[i].queueFlags & VK_QUEUE_TRANSFER_BIT) && !(qf[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT))) {
                tfam_ = i;
                break;
            }

        float prio = 1.0f;
        VkDeviceQueueCreateInfo qcis[2]{{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}, {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}};
        qcis[0].queueFamilyIndex = qfam_;
        qcis[0].queueCount = 1;
        qcis[0].pQueuePriorities = &prio;
        qcis[1].queueFamilyIndex = tfam_;
        qcis[1].queueCount = 1;
        qcis[1].pQueuePriorities = &prio;
        VkPhysicalDeviceCooperativeMatrixFeaturesKHR ecm{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
        ecm.cooperativeMatrix = VK_TRUE;
        VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        f13.subgroupSizeControl = VK_TRUE;
        f13.computeFullSubgroups = VK_TRUE;
        f13.pNext = coop_ ? &ecm : nullptr;
        VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        f12.bufferDeviceAddress = VK_TRUE;
        f12.shaderFloat16 = coop_ ? VK_TRUE : VK_FALSE;
        f12.pNext = &f13;
        VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        f2.features.shaderInt64 = VK_TRUE;
        f2.pNext = &f12;
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.pNext = &f2;
        dci.queueCreateInfoCount = tfam_ != UINT32_MAX ? 2 : 1;
        dci.pQueueCreateInfos = qcis;
        dci.enabledExtensionCount = (uint32_t)enable.size();
        dci.ppEnabledExtensionNames = enable.data();
        VK_CHECK(vkCreateDevice(phys_, &dci, nullptr, &dev_));
        vkGetDeviceQueue(dev_, qfam_, 0, &queue_);
        if (tfam_ != UINT32_MAX) vkGetDeviceQueue(dev_, tfam_, 0, &tqueue_);

        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = qfam_;
        VK_CHECK(vkCreateCommandPool(dev_, &pci, nullptr, &pool_));
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = pool_;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(dev_, &cai, &cmd_));
        VK_CHECK(vkAllocateCommandBuffers(dev_, &cai, &aux_));
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VK_CHECK(vkCreateFence(dev_, &fci, nullptr, &fence_));

        timestamp_ns_ = props_.limits.timestampPeriod;
        profile_ = std::getenv("QW_GPU_PROFILE") != nullptr;
        if (profile_) {
            VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
            qpi.queryCount = kMaxQueries;
            VK_CHECK(vkCreateQueryPool(dev_, &qpi, nullptr, &qpool_));
        }

        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 128};
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pcr;
        VK_CHECK(vkCreatePipelineLayout(dev_, &plci, nullptr, &layout_));
        for (auto& s : kShaders)
            if (coop_ || std::strncmp(s.name, "gemm_coop", 9) != 0) pipelines_[s.name] = make_pipeline(s);
        std::fprintf(stderr, "[vulkan] %s | subgroup %u-%u | shared mem %u KB | tensor cores (coop matrix f16): %s"
                             " | fp8 coop matrix: %s | copy engine: %s\n",
                     props_.deviceName, sscp.minSubgroupSize, sscp.maxSubgroupSize,
                     props_.limits.maxComputeSharedMemorySize / 1024, coop_ ? "yes" : "no",
                     coop_fp8_ ? "available" : "no", tfam_ != UINT32_MAX ? "yes" : "no");
    }
    bool coop() const { return coop_; }

    ~Gpu() {
        if (!dev_) return;
        vkDeviceWaitIdle(dev_);
        if (profile_) {
            std::vector<std::pair<double, std::string>> v;
            double total = 0;
            for (auto& [n, t] : prof_) { v.push_back({t.first, n}); total += t.first; }
            std::sort(v.rbegin(), v.rend());
            std::fprintf(stderr, "[gpu profile] total %.1f ms\n", total);
            for (auto& [t, n] : v) std::fprintf(stderr, "  %-28s %9.2f ms %8llu calls\n", n.c_str(), t, (unsigned long long)prof_[n].second);
            vkDestroyQueryPool(dev_, qpool_, nullptr);
        }
        for (auto& [n, p] : pipelines_) vkDestroyPipeline(dev_, p, nullptr);
        for (Buf* b : bufs_) {
            vkDestroyBuffer(dev_, b->buf, nullptr);
            vkFreeMemory(dev_, b->mem, nullptr);
            delete b;
        }
        vkDestroyPipelineLayout(dev_, layout_, nullptr);
        vkDestroyFence(dev_, fence_, nullptr);
        vkDestroyCommandPool(dev_, pool_, nullptr);
        vkDestroyDevice(dev_, nullptr);
        vkDestroyInstance(instance_, nullptr);
    }

    const char* device_name() const { return props_.deviceName; }

    // Free device-local memory we may still allocate (budget - usage), or heap size if unknown.
    VkDeviceSize device_free() const {
        int heap = -1;
        for (uint32_t i = 0; i < memprops_.memoryHeapCount; i++)
            if ((memprops_.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) &&
                (heap < 0 || memprops_.memoryHeaps[i].size > memprops_.memoryHeaps[heap].size))
                heap = (int)i;
        if (!has_budget_) return memprops_.memoryHeaps[heap].size - allocated_device_;
        VkPhysicalDeviceMemoryBudgetPropertiesEXT bp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
        VkPhysicalDeviceMemoryProperties2 mp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
        mp.pNext = &bp;
        vkGetPhysicalDeviceMemoryProperties2(phys_, &mp);
        return bp.heapBudget[heap] > bp.heapUsage[heap] ? bp.heapBudget[heap] - bp.heapUsage[heap] : 0;
    }

    // Returns nullptr instead of throwing when `may_fail` and the allocation does not succeed.
    Buf* create(VkDeviceSize size, Mem kind, bool may_fail = false) {
        Buf* b = new Buf;
        b->size = size;
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = size;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                    VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
        VK_CHECK(vkCreateBuffer(dev_, &bci, nullptr, &b->buf));
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(dev_, b->buf, &req);
        VkMemoryAllocateFlagsInfo fi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
        fi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.pNext = &fi;
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = memory_type(req.memoryTypeBits, kind);
        VkResult r = vkAllocateMemory(dev_, &mai, nullptr, &b->mem);
        if (r != VK_SUCCESS) {
            vkDestroyBuffer(dev_, b->buf, nullptr);
            delete b;
            if (may_fail) return nullptr;
            throw std::runtime_error("GPU memory allocation failed (" + std::to_string(size >> 20) + " MB)");
        }
        VK_CHECK(vkBindBufferMemory(dev_, b->buf, b->mem, 0));
        VkBufferDeviceAddressInfo ai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        ai.buffer = b->buf;
        b->addr = vkGetBufferDeviceAddress(dev_, &ai);
        if (kind != Mem::Device) VK_CHECK(vkMapMemory(dev_, b->mem, 0, VK_WHOLE_SIZE, 0, (void**)&b->map));
        else allocated_device_ += req.size;
        bufs_.push_back(b);
        return b;
    }

    void destroy(Buf* b) {
        vkDestroyBuffer(dev_, b->buf, nullptr);
        vkFreeMemory(dev_, b->mem, nullptr);
        bufs_.erase(std::find(bufs_.begin(), bufs_.end(), b));
        delete b;
    }

    // --- command recording -------------------------------------------------------------------
    void begin() {
        VK_CHECK(vkResetCommandBuffer(cmd_, 0));
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cmd_, &bi));
        if (profile_) {
            vkCmdResetQueryPool(cmd_, qpool_, 0, kMaxQueries);
            labels_.clear();
            vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool_, 0);
        }
    }
    // Timestamp after the preceding commands, attributed to `label` (profiling only).
    void mark(const char* label) {
        if (!profile_ || labels_.size() + 1 >= kMaxQueries) return;
        labels_.push_back(label);
        vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool_, (uint32_t)labels_.size());
    }
    void submit_wait() {
        submit();
        wait();
    }
    // Submit without waiting (the CPU can work meanwhile); wait() must follow before the next begin().
    void submit() {
        VK_CHECK(vkEndCommandBuffer(cmd_));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd_;
        VK_CHECK(vkQueueSubmit(queue_, 1, &si, fence_));
    }
    void wait() {
        VK_CHECK(vkWaitForFences(dev_, 1, &fence_, VK_TRUE, UINT64_MAX));
        VK_CHECK(vkResetFences(dev_, 1, &fence_));
        if (profile_ && !labels_.empty()) {
            std::vector<uint64_t> ts(labels_.size() + 1);
            vkGetQueryPoolResults(dev_, qpool_, 0, (uint32_t)ts.size(), ts.size() * 8, ts.data(), 8,
                                  VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            for (size_t i = 0; i < labels_.size(); i++) {
                auto& e = prof_[labels_[i]];
                e.first += (ts[i + 1] - ts[i]) * timestamp_ns_ * 1e-6;
                e.second++;
            }
        }
    }
    VkPipeline pipeline(const char* name) {
        auto it = pipelines_.find(name);
        if (it == pipelines_.end()) throw std::runtime_error(std::string("no pipeline ") + name);
        return it->second;
    }
    template <class PC>
    void dispatch(const char* name, const PC& pc, uint32_t gx, uint32_t gy = 1, uint32_t gz = 1, const char* label = nullptr) {
        static_assert(sizeof(PC) <= 128, "push constants too large");
        vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline(name));
        vkCmdPushConstants(cmd_, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PC), &pc);
        vkCmdDispatch(cmd_, gx, gy, gz);
        mark(label ? label : name);
    }
    template <class PC>
    void dispatch_indirect(const char* name, const PC& pc, Ptr args) {
        static_assert(sizeof(PC) <= 128, "push constants too large");
        vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline(name));
        vkCmdPushConstants(cmd_, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PC), &pc);
        vkCmdDispatchIndirect(cmd_, args.b->buf, args.off);
        mark(name);
    }
    // Full memory barrier between compute / transfer / indirect / host stages.
    void barrier() {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                           VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT |
                                 VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                             0, 1, &mb, 0, nullptr, 0, nullptr);
    }
    void copy(Buf* src, VkDeviceSize soff, Buf* dst, VkDeviceSize doff, VkDeviceSize size) {
        VkBufferCopy c{soff, doff, size};
        vkCmdCopyBuffer(cmd_, src->buf, dst->buf, 1, &c);
    }
    void update(Ptr dst, const void* data, VkDeviceSize size) {  // size <= 65536, multiple of 4
        vkCmdUpdateBuffer(cmd_, dst.b->buf, dst.off, size, data);
    }
    void fill_zero(Buf* b) { vkCmdFillBuffer(cmd_, b->buf, 0, VK_WHOLE_SIZE, 0); }

    // Independent copy-only submission (used by the uploader while the main command buffer is open).
    void aux_copies(Buf* src, const std::vector<std::pair<Buf*, VkBufferCopy>>& copies) {
        VK_CHECK(vkResetCommandBuffer(aux_, 0));
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(aux_, &bi));
        for (auto& [dst, c] : copies) vkCmdCopyBuffer(aux_, src->buf, dst->buf, 1, &c);
        VK_CHECK(vkEndCommandBuffer(aux_));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &aux_;
        VK_CHECK(vkQueueSubmit(queue_, 1, &si, fence_));
        VK_CHECK(vkWaitForFences(dev_, 1, &fence_, VK_TRUE, UINT64_MAX));
        VK_CHECK(vkResetFences(dev_, 1, &fence_));
    }

private:
    uint32_t memory_type(uint32_t bits, Mem kind) const {
        int best = -1, best_score = -1;
        for (uint32_t i = 0; i < memprops_.memoryTypeCount; i++) {
            if (!(bits & (1u << i))) continue;
            VkMemoryPropertyFlags f = memprops_.memoryTypes[i].propertyFlags;
            const bool dl = f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, hv = f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                       hc = f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, cached = f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
            int score;
            if (kind == Mem::Device) {
                if (!dl) continue;
                score = hv ? 1 : 2;
            } else {
                if (!hv || !hc) continue;
                score = dl ? 0 : 4;
                // readback wants CPU caching; GPU-read uploads prefer uncached (no snooping)
                if (kind == Mem::HostRead) score += cached ? 2 : 0;
                else score += cached ? 0 : 1;
            }
            if (score > best_score) { best = (int)i; best_score = score; }
        }
        if (best < 0) throw std::runtime_error("no suitable memory type");
        return (uint32_t)best;
    }

    VkPipeline make_pipeline(const EmbeddedShader& s) {
        std::vector<uint32_t> code((s.size + 3) / 4);
        std::memcpy(code.data(), s.data, s.size);
        VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smci.codeSize = s.size;
        smci.pCode = code.data();
        VkShaderModule mod;
        VK_CHECK(vkCreateShaderModule(dev_, &smci, nullptr, &mod));
        VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rss{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
        rss.requiredSubgroupSize = 32;
        VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpci.stage.pNext = &rss;
        cpci.stage.flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
        cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpci.stage.module = mod;
        cpci.stage.pName = "main";
        cpci.layout = layout_;
        VkPipeline p;
        VK_CHECK(vkCreateComputePipelines(dev_, VK_NULL_HANDLE, 1, &cpci, nullptr, &p));
        vkDestroyShaderModule(dev_, mod, nullptr);
        return p;
    }

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice phys_ = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties props_{};
    VkPhysicalDeviceMemoryProperties memprops_{};
    VkDevice dev_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t qfam_ = 0;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    VkCommandBuffer aux_ = VK_NULL_HANDLE;
    static constexpr uint32_t kMaxQueries = 4096;
    bool profile_ = false;
    float timestamp_ns_ = 1.0f;
    VkQueryPool qpool_ = VK_NULL_HANDLE;
    std::vector<const char*> labels_;
    std::unordered_map<std::string, std::pair<double, uint64_t>> prof_;  // label -> (ms, calls)
    VkFence fence_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    std::unordered_map<std::string, VkPipeline> pipelines_;
    std::vector<Buf*> bufs_;
    VkDeviceSize allocated_device_ = 0;
    bool has_budget_ = false;
    bool coop_ = false, coop_fp8_ = false;
    uint32_t tfam_ = UINT32_MAX;
    VkQueue tqueue_ = VK_NULL_HANDLE;
};

// Sub-allocates device-local memory out of 1 GiB buffers.
class Arena {
public:
    explicit Arena(Gpu& g) : g_(g) {}
    Ptr alloc(VkDeviceSize size) {
        size = (size + 255) & ~VkDeviceSize(255);
        if (size > CHUNK) return {g_.create(size, Mem::Device), 0};
        if (!cur_ || used_ + size > cur_->size) {
            cur_ = g_.create(CHUNK, Mem::Device);
            used_ = 0;
        }
        Ptr p{cur_, used_};
        used_ += size;
        return p;
    }

private:
    Gpu& g_;
    Buf* cur_ = nullptr;
    VkDeviceSize used_ = 0;
};

// Copies host data to device memory through a staging buffer (separate submissions).
class Uploader {
public:
    explicit Uploader(Gpu& g) : g_(g) { staging_ = g.create(STAGING, Mem::HostUpload); }
    void put(Ptr dst, const void* src, VkDeviceSize size) {
        const uint8_t* s = (const uint8_t*)src;
        while (size > 0) {
            if (used_ == STAGING) flush();
            VkDeviceSize n = std::min(size, STAGING - used_);
            std::memcpy(staging_->map + used_, s, n);
            pending_.push_back({dst.b, {used_, dst.off, n}});
            used_ += n;
            s += n;
            dst.off += n;
            size -= n;
        }
    }
    template <class F>
    void put_fn(Ptr dst, VkDeviceSize size, F&& fill) {  // fill(void* dst) writes `size` bytes; size <= STAGING
        if (used_ + size > STAGING) flush();
        fill(staging_->map + used_);
        pending_.push_back({dst.b, {used_, dst.off, size}});
        used_ += size;
    }
    void flush() {
        if (pending_.empty()) return;
        g_.aux_copies(staging_, pending_);
        pending_.clear();
        used_ = 0;
    }

private:
    Gpu& g_;
    Buf* staging_;
    VkDeviceSize used_ = 0;
    std::vector<std::pair<Buf*, VkBufferCopy>> pending_;
};

// ---------------------------------------------------------------------------------------------
// Push-constant layouts (must match the shaders, std430)
// ---------------------------------------------------------------------------------------------
struct PcMatFp8 { uint64_t w1, s1, w2, s2, x, y; uint32_t rows, cols, n, x_stride, y_stride, accumulate; };
struct PcGemm { uint64_t w1, s1, w2, s2, x, y; uint32_t rows, cols, n, x_stride, y_stride; };
struct PcExpert { uint64_t table, toks, x, y; uint32_t woff1, soff1, woff2, soff2, rows, cols, x_stride, y_stride, x_mode; };
struct PcMatBf16 { uint64_t w, x, y; uint32_t rows, cols, n, x_stride, y_stride; };
struct PcNorm { uint64_t x, delta, w, out; uint32_t dim; float eps; };
struct PcConv { uint64_t data, w, state; uint32_t n, channels; };
struct PcDelta { uint64_t qkv, a, b, neg_a, dt_bias, state, out; uint32_t n, conv_dim, key_dim, n_v_heads, ratio; };
struct PcGatedNorm { uint64_t o, z, w; uint32_t n_heads; float eps; uint32_t sigm; };
struct PcQkRope { uint64_t qg, k, v, q_out, kcache, vcache, qn, kn, rope; uint32_t pos, n_heads, n_kv, rope_dim; float eps; };
struct PcAttn { uint64_t q, kcache, vcache, part, sel; uint32_t pos, n_heads, n_kv, chunk, n_split; float scale; uint32_t ratio, topb; };
struct PcAttnCombine { uint64_t part, qg, out; uint32_t n_heads, n_split; };
struct PcRoute { uint64_t logits, sel, selw; uint32_t n_experts, top_k; };
struct PcGroup { uint64_t sel, selw, loc, entries, toks, comb, args; uint32_t n_sel, top_k, n_experts, chunk, gx_up, gx_down; };
struct PcCombine { uint64_t x, sgate, sy, ey, table; uint32_t dim, top_k, accumulate; };
struct PcCombW { uint64_t comb, selw; uint32_t n; };
// qwen4
struct PcGroupNorm { uint64_t x, w, out; uint32_t x_stride, out_stride, dim; float eps; };
struct PcHcMix { uint64_t u, hn, out; uint32_t dim, streams; };
struct PcHcOut { uint64_t x, y, inj; uint32_t dim, streams; };
struct PcEw { uint64_t a, b; uint32_t n, mode; float scale; };
struct PcPleGate { uint64_t kn, qn, val, gv; uint32_t dim, streams; };
struct PcPleConv { uint64_t gv, gvn, x, w, state; uint32_t n, channels; };
struct PcIdxPrep { uint64_t qk, iq, kraw, qn, rope; uint32_t pos, n_heads, rope_dim; float eps; };
struct PcIdxPool { uint64_t kraw, kblk, kn, rope; uint32_t b0, ratio, rope_dim; float eps; };
struct PcIdxScore { uint64_t iq, kblk, scores; uint32_t pos, ratio, n_heads, stride; };
struct PcIdxSel { uint64_t scores, sel; uint32_t pos, ratio, topb, stride; };

struct ExpertEntry { uint64_t base; uint32_t start, count; };  // matches `Entry` in the expert shaders
static_assert(sizeof(ExpertEntry) == 16, "ExpertEntry layout");

// GPU-side views of weights.
struct GQ { Ptr w, s; int rows = 0, cols = 0; };
struct GB { Ptr w; int rows = 0, cols = 0; };

// Dense projection on the GPU: FP8 (q.rows != 0) or BF16.
struct GL { GQ q; GB b; };

// qwen4 hyper-connection weights (inject.rows == 0 for the final mixer)
struct GHc { Ptr norm; GB down, up, inject; };

struct GLayer {
    Ptr in_norm, post_norm;                 // f32
    GL q, k, v, o; Ptr qn, kn;              // attention
    GL qkv, z, out; GB a, b; Ptr conv, neg_a, dt_bias, lnorm;  // deltanet
    GB router, sgate; GL sh_gate, sh_up, sh_down;
    GHc attn_hc, mlp_hc;                    // qwen4
    GB idx_qk; Ptr idx_qn, idx_kn;          // qwen4 indexer
};

uint32_t groups(uint32_t n, uint32_t per) { return (n + per - 1) / per; }

// ---------------------------------------------------------------------------------------------
class VulkanBackend final : public Backend {
public:
    VulkanBackend(const Config& c, const ModelWeights& w, int max_ctx, const VulkanOptions& opt)
        : c_(c), w_(w), max_ctx_(max_ctx), opt_(opt), arena_(gpu_), up_(gpu_) {
        upload_weights();
        alloc_state();
        setup_experts();
        reset();
    }

    ~VulkanBackend() override {
        std::fprintf(stderr, "[vulkan] expert lookups: %llu cached, %llu %s; %llu cache fills\n", (unsigned long long)hits_,
                     (unsigned long long)misses_, hybrid_ ? "computed on the CPU" : "read from host memory",
                     (unsigned long long)fills_);
        save_stats();
    }

    void reset() override {
        pos_ = 0;
        gpu_.begin();
        gpu_.fill_zero(conv_state_);
        gpu_.fill_zero(rec_state_);
        if (ple_state_) gpu_.fill_zero(ple_state_);
        gpu_.submit_wait();
        hist_.assign(std::max(c_.ngram - 1, 0), c_.ple_eos);
    }
    int pos() const override { return pos_; }
    int max_ctx() const override { return max_ctx_; }
    const char* name() const override { return "vulkan"; }

    void save_state() override {
        gpu_.begin();
        gpu_.copy(conv_state_, 0, conv_saved_, 0, conv_state_->size);
        gpu_.copy(rec_state_, 0, rec_saved_, 0, rec_state_->size);
        if (ple_state_) gpu_.copy(ple_state_, 0, ple_saved_, 0, ple_state_->size);
        gpu_.submit_wait();
        saved_pos_ = pos_;
        saved_hist_ = hist_;
    }
    void restore_state() override {
        gpu_.begin();
        gpu_.copy(conv_saved_, 0, conv_state_, 0, conv_state_->size);
        gpu_.copy(rec_saved_, 0, rec_state_, 0, rec_state_->size);
        if (ple_state_) gpu_.copy(ple_saved_, 0, ple_state_, 0, ple_state_->size);
        gpu_.submit_wait();
        pos_ = saved_pos_;
        hist_ = saved_hist_;
    }

    void forward(const int* tokens, int n, float* logits) override {
        if (pos_ + n > max_ctx_) throw std::runtime_error("context overflow");
        for (int done = 0; done < n;) {
            const int m = std::min(NMAX, n - done);
            forward_chunk(tokens + done, m, done + m == n ? logits : nullptr);
            done += m;
        }
    }

private:
    const Config& c_;
    const ModelWeights& w_;
    int max_ctx_;
    VulkanOptions opt_;
    int pos_ = 0, saved_pos_ = 0;
    Gpu gpu_;
    Arena arena_;
    Uploader up_;

    std::vector<GLayer> L_;
    GB lm_head_;
    Ptr final_norm_, rope_;
    std::vector<int> slot_;  // per layer: index into full-attn or linear-attn state arrays

    // activations (device local)
    Ptr x_, xn_, mix_, qkv_, z_, a_, b_, core_, qg_, kb_, vb_, qo_, att_, part_, sact_, sy_, sgate_, act_, ey_;
    Ptr rlogits_, selw_, toks_, comb_, entries_, args_, sel_, logits_;
    Buf* kcache_ = nullptr;
    Buf* vcache_ = nullptr;
    Buf* conv_state_ = nullptr;
    Buf* rec_state_ = nullptr;
    Buf* conv_saved_ = nullptr;
    Buf* rec_saved_ = nullptr;
    // host-visible
    Buf* hx_ = nullptr;     // embeddings of the current chunk
    Buf* hsel_ = nullptr;   // selected expert ids, [layer][NMAX * top_k]
    Buf* hlog_ = nullptr;   // final logits
    // qwen4: x_ holds hc streams [n, hc * hidden]; xn_ the mixed block input; mix_ the block output
    int HH_ = 0;
    GHc mixer_;
    GB ple_key_, ple_val_; Ptr ple_nk_, ple_nq_, ple_nc_, ple_conv_;
    Ptr hn_, hd_, hu_, inj_, su_, emb_, pk_, pv_, iqk_, iq_, scores_, isel_;
    Buf* hemb_ = nullptr;       // PLE n-gram embeddings computed on the CPU
    Buf* ple_state_ = nullptr;  // dilated conv history [(K-1)*dil][hc*hidden]
    Buf* ple_saved_ = nullptr;
    Buf* kraw_ = nullptr;       // indexer raw keys [full layer][pos][idx_dim]
    Buf* kblk_ = nullptr;       // indexer pooled block keys [full layer][block][idx_dim]
    int nblk_ = 0;              // blocks per layer in kblk_
    std::vector<int> hist_, saved_hist_;
    std::unique_ptr<ThreadPool> ple_pool_;  // CPU threads for the PLE lookups

    // experts
    struct Slot { Buf* buf = nullptr; VkDeviceSize off = 0; int key = -1; uint64_t tick = 0; };
    std::vector<Slot> slots_;               // VRAM cache
    std::vector<int> where_;                // key (layer * n_experts + expert) -> slot or -1
    std::vector<Buf*> host_bufs_;           // pinned copies of all experts
    VkDeviceSize host_per_buf_ = 0;
    bool pinned_ = false;
    bool hybrid_ = false;                   // CPU computes cache misses (see header comment)
    std::unique_ptr<ThreadPool> cpu_;
    Buf* hxn_ = nullptr;                    // hybrid: normalized activations of the layer (GPU -> CPU)
    Buf* htab_ = nullptr;                   // hybrid: entries | toks | comb tables written by the CPU
    Buf* hcy_ = nullptr;                    // hybrid: expert outputs computed on the CPU (CPU -> GPU)
    VkDeviceSize htab_toks_ = 0, htab_comb_ = 0;
    uint64_t cpu_rows_ = 0;
    Ptr loc_;                               // [layers * n_experts] uint64 address the shaders read from
    std::vector<int> pending_;              // keys to bring into the cache at the start of the next pass
    std::vector<uint64_t> usage_;           // selection counts per key (persisted)
    VkDeviceSize slot_size_ = 0, off_up_ = 0, off_down_ = 0, off_sg_ = 0, off_su_ = 0, off_sd_ = 0;
    uint64_t tick_ = 0, hits_ = 0, misses_ = 0, fills_ = 0;

    // ---- setup ----------------------------------------------------------------------------
    GQ up_q(const QMat& m) {
        GQ g;
        g.rows = m.rows;
        g.cols = m.cols;
        g.w = arena_.alloc(m.weight_bytes());
        g.s = arena_.alloc(m.scale_count() * 2);
        up_.put(g.w, m.w, m.weight_bytes());
        up_.put(g.s, m.scale, m.scale_count() * 2);
        return g;
    }
    GB up_b(const BMat& m) {
        GB g;
        g.rows = m.rows;
        g.cols = m.cols;
        g.w = arena_.alloc((size_t)m.rows * m.cols * 2);
        up_.put(g.w, m.w, (size_t)m.rows * m.cols * 2);
        return g;
    }
    Ptr up_f32(const std::vector<float>& v) {
        Ptr p = arena_.alloc(v.size() * 4);
        up_.put(p, v.data(), v.size() * 4);
        return p;
    }
    GL up_lin(const Lin& m) {
        GL g;
        if (m.fp8()) g.q = up_q(m.q);
        else g.b = up_b(m.b);
        return g;
    }
    GHc up_hc(const HyperConn& h) {
        GHc g;
        g.norm = up_f32(to_f32(h.norm));
        g.down = up_b(h.down);
        g.up = up_b(h.up);
        if (h.inject.rows) g.inject = up_b(h.inject);
        return g;
    }
    static std::vector<float> to_f32(const BMat& m) {
        std::vector<float> v((size_t)m.rows * m.cols);
        for (size_t i = 0; i < v.size(); i++) v[i] = m.at(i);
        return v;
    }

    void upload_weights() {
        lm_head_ = up_b(w_.lm_head);
        if (c_.qwen4) {
            mixer_ = up_hc(w_.mixer);
            if (c_.ple_layer >= 0) {
                const Ple& P = w_.ple;
                ple_key_ = up_b(P.key_proj);
                ple_val_ = up_b(P.value_proj);
                ple_nk_ = up_f32(to_f32(P.norm_key));
                ple_nq_ = up_f32(to_f32(P.norm_query));
                ple_nc_ = up_f32(to_f32(P.norm_conv));
                ple_conv_ = up_f32(to_f32(P.conv));
            }
        } else {
            final_norm_ = up_f32(to_f32(w_.final_norm));
        }
        int nf = 0, nl = 0;
        for (int l = 0; l < c_.n_layers; l++) {
            const LayerWeights& W = w_.layers[l];
            GLayer g;
            if (c_.qwen4) {
                g.attn_hc = up_hc(W.attn_hc);
                g.mlp_hc = up_hc(W.mlp_hc);
            } else {
                g.in_norm = up_f32(to_f32(W.input_norm));
                g.post_norm = up_f32(to_f32(W.post_norm));
            }
            if (W.full_attn) {
                slot_.push_back(nf++);
                g.q = up_lin(W.q_proj); g.k = up_lin(W.k_proj); g.v = up_lin(W.v_proj); g.o = up_lin(W.o_proj);
                g.qn = up_f32(to_f32(W.q_norm));
                g.kn = up_f32(to_f32(W.k_norm));
                if (c_.qwen4) {
                    g.idx_qk = up_b(W.idx_qk);
                    g.idx_qn = up_f32(to_f32(W.idx_qn));
                    g.idx_kn = up_f32(to_f32(W.idx_kn));
                }
            } else {
                slot_.push_back(nl++);
                g.qkv = up_lin(W.in_qkv); g.z = up_lin(W.in_z); g.out = up_lin(W.out_proj);
                g.a = up_b(W.in_a); g.b = up_b(W.in_b);
                g.conv = up_f32(to_f32(W.conv1d));
                std::vector<float> na = to_f32(W.a_log);
                for (float& v : na) v = -std::exp(v);
                g.neg_a = up_f32(na);
                g.dt_bias = up_f32(to_f32(W.dt_bias));
                g.lnorm = up_f32(to_f32(W.lin_norm));
            }
            g.router = up_b(W.router);
            g.sgate = up_b(W.shared_gate);
            g.sh_gate = up_lin(W.sh_gate); g.sh_up = up_lin(W.sh_up); g.sh_down = up_lin(W.sh_down);
            L_.push_back(g);
        }
        // rope table [max_ctx][rope_dim/2] of (cos, sin), computed like the CPU path
        const int half = c_.rope_dim / 2;
        std::vector<float> rope((size_t)max_ctx_ * c_.rope_dim);
        for (int p = 0; p < max_ctx_; p++)
            for (int i = 0; i < half; i++) {
                float inv = (float)(1.0 / std::pow((double)c_.rope_theta, (2.0 * i) / c_.rope_dim));
                float ang = (float)p * inv;
                rope[(size_t)p * c_.rope_dim + 2 * i] = std::cos(ang);
                rope[(size_t)p * c_.rope_dim + 2 * i + 1] = std::sin(ang);
            }
        rope_ = up_f32(rope);
        up_.flush();
    }

    void alloc_state() {
        const int H = c_.hidden, TK = c_.top_k;
        const int n_full = (int)std::count(c_.is_full_attn.begin(), c_.is_full_attn.end(), true);
        const int n_lin = c_.n_layers - n_full;
        auto act = [&](size_t floats) { return arena_.alloc(floats * 4); };
        HH_ = c_.hc * H;
        x_ = act((size_t)NMAX * HH_);
        xn_ = act((size_t)NMAX * H);
        mix_ = act((size_t)NMAX * H);
        qkv_ = act((size_t)NMAX * c_.conv_dim());
        z_ = act((size_t)NMAX * c_.lin_value_dim());
        a_ = act((size_t)NMAX * c_.lin_v_heads);
        b_ = act((size_t)NMAX * c_.lin_v_heads);
        core_ = act((size_t)NMAX * c_.lin_value_dim());
        qg_ = act((size_t)NMAX * c_.n_heads * c_.head_dim * 2);
        kb_ = act((size_t)NMAX * c_.n_kv_heads * c_.head_dim);
        vb_ = act((size_t)NMAX * c_.n_kv_heads * c_.head_dim);
        qo_ = act((size_t)NMAX * c_.n_heads * c_.head_dim);
        att_ = act((size_t)NMAX * c_.n_heads * c_.head_dim);
        part_ = act((size_t)ATT_PART_MAX * c_.n_heads * (c_.head_dim + 2));
        sact_ = act((size_t)NMAX * c_.shared_inter);
        sy_ = act((size_t)NMAX * H);
        sgate_ = act(NMAX);
        act_ = act((size_t)NMAX * TK * c_.moe_inter);
        ey_ = act((size_t)NMAX * TK * H);
        rlogits_ = act((size_t)NMAX * c_.n_experts);
        selw_ = act((size_t)NMAX * TK);
        toks_ = act((size_t)NMAX * TK);
        comb_ = act((size_t)NMAX * TK * 2);
        entries_ = arena_.alloc((VkDeviceSize)NMAX * TK * sizeof(ExpertEntry));
        args_ = arena_.alloc(64);
        sel_ = arena_.alloc((VkDeviceSize)c_.n_layers * NMAX * TK * 4);
        logits_ = arena_.alloc((VkDeviceSize)c_.vocab * 4);
        const VkDeviceSize kv = (VkDeviceSize)n_full * max_ctx_ * c_.n_kv_heads * c_.head_dim * 4;
        kcache_ = gpu_.create(std::max<VkDeviceSize>(kv, 256), Mem::Device);
        vcache_ = gpu_.create(std::max<VkDeviceSize>(kv, 256), Mem::Device);
        const VkDeviceSize cs = std::max<VkDeviceSize>((VkDeviceSize)n_lin * (c_.conv_kernel - 1) * c_.conv_dim() * 4, 256);
        const VkDeviceSize rs = std::max<VkDeviceSize>((VkDeviceSize)n_lin * c_.lin_v_heads * c_.lin_k_dim * c_.lin_v_dim * 4, 256);
        conv_state_ = gpu_.create(cs, Mem::Device);
        rec_state_ = gpu_.create(rs, Mem::Device);
        conv_saved_ = gpu_.create(cs, Mem::Device);
        rec_saved_ = gpu_.create(rs, Mem::Device);
        hx_ = gpu_.create((VkDeviceSize)NMAX * HH_ * 4, Mem::HostUpload);
        if (c_.qwen4) {
            hn_ = act((size_t)NMAX * HH_);
            hu_ = act((size_t)NMAX * HH_);
            hd_ = act((size_t)NMAX * c_.hc_rank);
            inj_ = act((size_t)NMAX * c_.hc);
            su_ = act((size_t)NMAX * c_.shared_inter);
            if (c_.ple_layer >= 0) {
                emb_ = act((size_t)NMAX * c_.ple_dim);
                pk_ = act((size_t)NMAX * HH_);
                pv_ = act((size_t)NMAX * H);
                hemb_ = gpu_.create((VkDeviceSize)NMAX * c_.ple_dim * 4, Mem::HostUpload);
                const VkDeviceSize ps = (VkDeviceSize)(c_.ple_conv - 1) * c_.ngram * HH_ * 4;
                ple_state_ = gpu_.create(ps, Mem::Device);
                ple_saved_ = gpu_.create(ps, Mem::Device);
                ple_pool_ = std::make_unique<ThreadPool>(0);
            }
            const int D = c_.idx_dim;
            nblk_ = max_ctx_ / c_.idx_ratio + 1;
            iqk_ = act((size_t)NMAX * (c_.idx_heads + 1) * D);
            iq_ = act((size_t)NMAX * c_.idx_heads * D);
            scores_ = act((size_t)NMAX * nblk_);
            isel_ = arena_.alloc((VkDeviceSize)NMAX * (c_.idx_budget / c_.idx_ratio + 1) * 4);
            kraw_ = gpu_.create(std::max<VkDeviceSize>((VkDeviceSize)n_full * max_ctx_ * D * 4, 256), Mem::Device);
            kblk_ = gpu_.create(std::max<VkDeviceSize>((VkDeviceSize)n_full * nblk_ * D * 4, 256), Mem::Device);
        }
        hsel_ = gpu_.create((VkDeviceSize)c_.n_layers * NMAX * TK * 4, Mem::HostRead);
        hlog_ = gpu_.create((VkDeviceSize)c_.vocab * 4, Mem::HostRead);
    }

    // ---- experts ------------------------------------------------------------------------------
    size_t n_keys() const { return (size_t)c_.n_layers * c_.n_experts; }
    const Expert& expert_of(int key) const { return w_.layers[key / c_.n_experts].experts[key % c_.n_experts]; }
    Ptr host_ptr(int key) const {
        return {host_bufs_[key / host_per_buf_], (VkDeviceSize)(key % host_per_buf_) * slot_size_};
    }
    uint64_t home_addr(int key) const { return pinned_ ? host_ptr(key).addr() : 0; }

    void write_expert(const Expert& ex, uint8_t* d) const {
        std::memcpy(d, ex.gate.w, ex.gate.weight_bytes());
        std::memcpy(d + off_up_, ex.up.w, ex.up.weight_bytes());
        std::memcpy(d + off_down_, ex.down.w, ex.down.weight_bytes());
        std::memcpy(d + off_sg_, ex.gate.scale, ex.gate.scale_count() * 2);
        std::memcpy(d + off_su_, ex.up.scale, ex.up.scale_count() * 2);
        std::memcpy(d + off_sd_, ex.down.scale, ex.down.scale_count() * 2);
    }

    void setup_experts() {
        const Expert& e0 = w_.layers[0].experts[0];
        auto align = [](VkDeviceSize v) { return (v + 255) & ~VkDeviceSize(255); };
        VkDeviceSize o = align(e0.gate.weight_bytes());
        off_up_ = o; o = align(o + e0.up.weight_bytes());
        off_down_ = o; o = align(o + e0.down.weight_bytes());
        off_sg_ = o; o = align(o + e0.gate.scale_count() * 2);
        off_su_ = o; o = align(o + e0.up.scale_count() * 2);
        off_sd_ = o; o = align(o + e0.down.scale_count() * 2);
        slot_size_ = o;
        usage_.assign(n_keys(), 0);
        load_stats();

        // VRAM cache slots
        VkDeviceSize budget;
        if (opt_.expert_cache_mb) budget = (VkDeviceSize)opt_.expert_cache_mb << 20;
        else {
            const VkDeviceSize free = gpu_.device_free(), margin = 1536ull << 20;  // driver / display headroom
            budget = free > margin ? free - margin : 0;
        }
        size_t want = std::min<size_t>(n_keys(), budget / slot_size_);
        want = std::max<size_t>(want, std::min<size_t>(n_keys(), c_.n_experts));  // at least one layer's worth
        const size_t per_chunk = CHUNK / slot_size_;
        while (slots_.size() < want) {
            const size_t k = std::min(per_chunk, want - slots_.size());
            Buf* b = gpu_.create(k * slot_size_, Mem::Device, true);
            if (!b) break;
            for (size_t i = 0; i < k; i++) slots_.push_back({b, i * slot_size_, -1, 0});
        }
        where_.assign(n_keys(), -1);
        // Misses are read by the GPU from pinned host memory when it fits (fast on PCIe 4/5 machines: measured
        // faster than computing them on the CPU here). If pinning fails, or with --hybrid, the CPU computes them.
        hybrid_ = opt_.hybrid > 0;
        if (!hybrid_) {
            // pinned host copies of every expert
            host_per_buf_ = CHUNK / slot_size_;
            if (opt_.pin_experts) {
                const size_t nbuf = (n_keys() + host_per_buf_ - 1) / host_per_buf_;
                pinned_ = true;
                for (size_t i = 0; i < nbuf; i++) {
                    const size_t cnt = std::min<size_t>(host_per_buf_, n_keys() - i * host_per_buf_);
                    Buf* b = gpu_.create(cnt * slot_size_, Mem::HostUpload, true);
                    if (!b) {  // not enough pinnable memory: release what we got
                        pinned_ = false;
                        for (Buf* x : host_bufs_) gpu_.destroy(x);
                        host_bufs_.clear();
                        break;
                    }
                    host_bufs_.push_back(b);
                }
                if (pinned_) {
                    auto t0 = std::chrono::steady_clock::now();
                    const int nth = std::max(1, std::min(16, (int)std::thread::hardware_concurrency()));
                    std::vector<std::thread> th;
                    for (int t = 0; t < nth; t++)
                        th.emplace_back([&, t] {
                            for (size_t k = t; k < n_keys(); k += nth) write_expert(expert_of((int)k), host_bufs_[k / host_per_buf_]->map + (k % host_per_buf_) * slot_size_);
                        });
                    for (auto& x : th) x.join();
                    std::fprintf(stderr, "[vulkan] pinned %zu experts (%.1f GB) in host memory in %.1fs\n", n_keys(),
                                 n_keys() * slot_size_ / 1073741824.0,
                                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
                } else {
                    std::fprintf(stderr, "[vulkan] could not pin %.1f GB of experts in host memory\n", n_keys() * slot_size_ / 1073741824.0);
                }
            }
            if (!pinned_ && opt_.hybrid < 0) hybrid_ = true;
            if (!pinned_ && !hybrid_ && slots_.size() < (size_t)c_.n_experts) throw std::runtime_error("not enough VRAM for the expert cache");
        }
        if (hybrid_) {
            const int H = c_.hidden, TK = c_.top_k;
            cpu_ = std::make_unique<ThreadPool>(0);
            hxn_ = gpu_.create((VkDeviceSize)NMAX * H * 4, Mem::HostRead);
            hcy_ = gpu_.create((VkDeviceSize)NMAX * TK * H * 4, Mem::HostUpload);
            const VkDeviceSize ne = (VkDeviceSize)NMAX * TK;
            htab_toks_ = (ne * sizeof(ExpertEntry) + 255) & ~VkDeviceSize(255);
            htab_comb_ = (htab_toks_ + ne * 4 + 255) & ~VkDeviceSize(255);
            htab_ = gpu_.create(htab_comb_ + ne * 8, Mem::HostUpload);
            std::fprintf(stderr, "[vulkan] hybrid mode: GPU runs cached experts, %d CPU threads compute the rest\n", cpu_->size());
        }
        std::fprintf(stderr, "[vulkan] expert cache: %zu slots x %.2f MB = %.1f GB (%.0f%% of %zu experts)\n",
                     slots_.size(), slot_size_ / 1048576.0, slots_.size() * slot_size_ / 1073741824.0,
                     100.0 * slots_.size() / n_keys(), n_keys());

        // location table: everything starts at its pinned home, then preload overwrites cached entries
        loc_ = arena_.alloc(n_keys() * 8);
        preload();
    }

    // Fill the cache with the most used experts (persisted counts), or evenly across layers.
    void preload() {
        std::vector<int> order(n_keys());
        std::iota(order.begin(), order.end(), 0);
        // stable: equal counts keep "expert-major" order (expert 0 of all layers, then expert 1, ...)
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            const int ea = a % c_.n_experts, eb = b % c_.n_experts;
            return usage_[a] != usage_[b] ? usage_[a] > usage_[b] : ea < eb;
        });
        const size_t n = std::min(slots_.size(), order.size());
        auto t0 = std::chrono::steady_clock::now();
        if (pinned_) {
            for (size_t i = 0; i < n; i += 256) {  // batched GPU copies from the pinned copies
                gpu_.begin();
                for (size_t k = i; k < std::min(n, i + 256); k++) {
                    const Ptr h = host_ptr(order[k]);
                    gpu_.copy(h.b, h.off, slots_[k].buf, slots_[k].off, slot_size_);
                }
                gpu_.submit_wait();
            }
        } else {
            for (size_t k = 0; k < n; k++)
                up_.put_fn({slots_[k].buf, slots_[k].off}, slot_size_, [&](void* d) { write_expert(expert_of(order[k]), (uint8_t*)d); });
            up_.flush();
        }
        for (size_t k = 0; k < n; k++) {
            slots_[k].key = order[k];
            where_[order[k]] = (int)k;
        }
        std::vector<uint64_t> loc(n_keys());
        for (size_t k = 0; k < n_keys(); k++) loc[k] = where_[k] >= 0 ? slot_addr(where_[k]) : home_addr((int)k);
        up_.put(loc_, loc.data(), loc.size() * 8);
        up_.flush();
        std::fprintf(stderr, "[vulkan] preloaded %zu experts (%s) in %.1fs\n", n,
                     std::any_of(usage_.begin(), usage_.end(), [](uint64_t u) { return u > 0; }) ? "by usage stats" : "uniform",
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }

    // cache fills per pass, by bytes: big experts would otherwise eat the PCIe bandwidth the misses need
    int max_fills() const { return std::max(1, std::min(MAX_UPLOADS, (int)((size_t)FILL_MB << 20) / (int)slot_size_)); }

    uint64_t slot_addr(int s) const { return slots_[s].buf->addr + slots_[s].off; }

    // Picks the least recently used slot not used in the current pass.
    int victim() {
        int v = -1;
        for (int i = 0; i < (int)slots_.size(); i++) {
            if (slots_[i].key < 0) return i;
            if (slots_[i].tick == tick_) continue;
            if (v < 0 || slots_[i].tick < slots_[v].tick) v = i;
        }
        return v;
    }

    // Records (into the open command buffer) moving `key` into a cache slot; returns false if none free.
    // With pinned memory the copy is GPU-side; otherwise data goes through the staging uploader.
    bool fill_slot(int key) {
        const int s = victim();
        if (s < 0) return false;
        Slot& sl = slots_[s];
        if (sl.key >= 0) {
            where_[sl.key] = -1;
            const uint64_t home = home_addr(sl.key);
            gpu_.update({loc_.b, loc_.off + (VkDeviceSize)sl.key * 8}, &home, 8);
        }
        if (pinned_) {
            const Ptr h = host_ptr(key);
            gpu_.copy(h.b, h.off, sl.buf, sl.off, slot_size_);
        } else {
            up_.put_fn({sl.buf, sl.off}, slot_size_, [&](void* d) { write_expert(expert_of(key), (uint8_t*)d); });
        }
        const uint64_t a = slot_addr(s);
        gpu_.update({loc_.b, loc_.off + (VkDeviceSize)key * 8}, &a, 8);
        sl.key = key;
        sl.tick = tick_;
        where_[key] = s;
        fills_++;
        return true;
    }

    // After a pass: count usage, refresh LRU ticks, queue misses for the next pass.
    void account(int l, int n) {
        const uint32_t* sel = (const uint32_t*)(hsel_->map + (size_t)l * NMAX * c_.top_k * 4);
        for (int i = 0; i < n * c_.top_k; i++) {
            const int key = l * c_.n_experts + (int)sel[i];
            usage_[key]++;
            if (where_[key] >= 0) {
                hits_++;
                slots_[where_[key]].tick = tick_;
            } else {
                misses_++;
                if (pinned_ && (int)pending_.size() < max_fills() &&
                    std::find(pending_.begin(), pending_.end(), key) == pending_.end())
                    pending_.push_back(key);
            }
        }
    }

    void load_stats() {
        if (opt_.stats_path.empty()) return;
        std::ifstream f(opt_.stats_path, std::ios::binary);
        if (!f) return;
        std::vector<uint64_t> v(n_keys());
        if (f.read((char*)v.data(), v.size() * 8)) usage_ = v;
    }
    void save_stats() {
        if (opt_.stats_path.empty()) return;
        std::ofstream f(opt_.stats_path, std::ios::binary);
        f.write((const char*)usage_.data(), usage_.size() * 8);
    }

    // ---- recording helpers --------------------------------------------------------------------
    void mat_q(const GQ& m, Ptr x, Ptr y, int n) {
        if (n >= GEMM_MIN) {
            PcGemm pc{m.w.addr(), m.s.addr(), 0, 0, x.addr(), y.addr(), (uint32_t)m.rows, (uint32_t)m.cols, (uint32_t)n,
                      (uint32_t)m.cols, (uint32_t)m.rows};
            gpu_.dispatch(gpu_.coop() ? "gemm_coop" : "gemm_fp8", pc, groups(m.rows, 64), groups(n, 64));
            return;
        }
        PcMatFp8 pc{m.w.addr(), m.s.addr(), 0, 0, x.addr(), y.addr(), (uint32_t)m.rows, (uint32_t)m.cols, (uint32_t)n,
                    (uint32_t)m.cols, (uint32_t)m.rows, 0};
        gpu_.dispatch("matmul_fp8", pc, groups(m.rows, 8), groups(n, NT));
    }
    void mat_swiglu(const GQ& g, const GQ& u, Ptr x, Ptr y, int n) {
        if (n >= GEMM_MIN) {
            PcGemm pc{g.w.addr(), g.s.addr(), u.w.addr(), u.s.addr(), x.addr(), y.addr(), (uint32_t)g.rows,
                      (uint32_t)g.cols, (uint32_t)n, (uint32_t)g.cols, (uint32_t)g.rows};
            gpu_.dispatch(gpu_.coop() ? "gemm_coop_swiglu" : "gemm_fp8_swiglu", pc, groups(g.rows, 64), groups(n, 64));
            return;
        }
        PcMatFp8 pc{g.w.addr(), g.s.addr(), u.w.addr(), u.s.addr(), x.addr(), y.addr(), (uint32_t)g.rows,
                    (uint32_t)g.cols, (uint32_t)n, (uint32_t)g.cols, (uint32_t)g.rows, 0};
        gpu_.dispatch("matmul_fp8_swiglu", pc, groups(g.rows, 8), groups(n, NT));
    }
    void mat_b(const GB& m, uint64_t x, uint64_t y, int n, int x_stride, const char* label = nullptr) {
        PcMatBf16 pc{m.w.addr(), x, y, (uint32_t)m.rows, (uint32_t)m.cols, (uint32_t)n, (uint32_t)x_stride, (uint32_t)m.rows};
        gpu_.dispatch("matmul_bf16", pc, groups(m.rows, 8), groups(n, NT), 1, label);
    }
    // y = x W^T for an FP8 or BF16 projection (x rows are cols wide)
    void lin(const GL& m, Ptr x, Ptr y, int n) {
        if (m.q.rows) mat_q(m.q, x, y, n);
        else mat_b(m.b, x.addr(), y.addr(), n, m.b.cols);
    }
    void ew(Ptr a, Ptr b, uint32_t count, uint32_t mode, float scale) {
        PcEw pc{a.addr(), b.addr(), count, mode, scale};
        gpu_.dispatch("elementwise", pc, groups(count, 256));
    }
    void group_norm(Ptr x, uint32_t x_stride, Ptr w, Ptr out, uint32_t out_stride, int n) {
        PcGroupNorm pc{x.addr(), w.addr(), out.addr(), x_stride, out_stride, (uint32_t)c_.hidden, c_.rms_eps};
        gpu_.dispatch("group_norm", pc, n, c_.hc);
    }

    // qwen4 hyper-connection input: normalize the streams of x, mix them into xn_, inject logits -> inj_.
    void hc_in(const GHc& h, Ptr x, int n) {
        const int S = c_.hc;
        group_norm(x, HH_, h.norm, hn_, HH_, n);
        gpu_.barrier();
        mat_b(h.down, hn_.addr(), hd_.addr(), n, HH_);
        if (h.inject.rows) mat_b(h.inject, hn_.addr(), inj_.addr(), n, HH_);
        gpu_.barrier();
        ew(hd_, hd_, (uint32_t)(n * c_.hc_rank), 0, 1.0f / S);
        gpu_.barrier();
        mat_b(h.up, hd_.addr(), hu_.addr(), n, c_.hc_rank);
        gpu_.barrier();
        PcHcMix pm{hu_.addr(), hn_.addr(), xn_.addr(), (uint32_t)c_.hidden, (uint32_t)S};
        gpu_.dispatch("hc_mix", pm, groups(c_.hidden, 256), n);
        gpu_.barrier();
    }
    // streams += block output (mix_) * injection weights
    void hc_out(int n) {
        PcHcOut po{x_.addr(), mix_.addr(), inj_.addr(), (uint32_t)c_.hidden, (uint32_t)c_.hc};
        gpu_.dispatch("hc_out", po, groups(c_.hidden, 256), n);
        gpu_.barrier();
    }
    // qwen4 PLE: n-gram embeddings (uploaded in emb_) gated into every stream, plus the dilated conv.
    void ple(int n) {
        const int H = c_.hidden, S = c_.hc;
        mat_b(ple_key_, emb_.addr(), pk_.addr(), n, c_.ple_dim);
        mat_b(ple_val_, emb_.addr(), pv_.addr(), n, c_.ple_dim);
        group_norm(x_, HH_, ple_nq_, hn_, HH_, n);
        gpu_.barrier();
        group_norm(pk_, HH_, ple_nk_, pk_, HH_, n);
        gpu_.barrier();
        PcPleGate pg{pk_.addr(), hn_.addr(), pv_.addr(), hu_.addr(), (uint32_t)H, (uint32_t)S};
        gpu_.dispatch("ple_gate", pg, n, S);
        gpu_.barrier();
        group_norm(hu_, HH_, ple_nc_, pk_, HH_, n);
        gpu_.barrier();
        PcPleConv pc{hu_.addr(), pk_.addr(), x_.addr(), ple_conv_.addr(), ple_state_->addr, (uint32_t)n, (uint32_t)HH_};
        gpu_.dispatch("ple_conv", pc, groups(HH_, 256));
        gpu_.barrier();
    }
    // qwen4 indexer: queries / raw keys, pooled keys of completed blocks, and (past the budget) the top-k
    // block selection per token. Returns the selection buffer address, or 0 when every position is attended.
    uint64_t indexer(int l, int n) {
        const GLayer& g = L_[l];
        const int D = c_.idx_dim, R = c_.idx_ratio, TOPB = c_.idx_budget / R;
        const uint64_t kr = kraw_->addr + (uint64_t)slot_[l] * max_ctx_ * D * 4;
        const uint64_t kb = kblk_->addr + (uint64_t)slot_[l] * nblk_ * D * 4;
        mat_b(g.idx_qk, xn_.addr(), iqk_.addr(), n, c_.hidden);
        gpu_.barrier();
        PcIdxPrep pp{iqk_.addr(), iq_.addr(), kr, g.idx_qn.addr(), rope_.addr(), (uint32_t)pos_, (uint32_t)c_.idx_heads,
                     (uint32_t)c_.rope_dim, c_.rms_eps};
        gpu_.dispatch("idx_prep", pp, n, c_.idx_heads + 1);
        gpu_.barrier();
        const int b0 = pos_ / R, b1 = (pos_ + n) / R;
        if (b1 > b0) {
            PcIdxPool pl{kr, kb, g.idx_kn.addr(), rope_.addr(), (uint32_t)b0, (uint32_t)R, (uint32_t)c_.rope_dim, c_.rms_eps};
            gpu_.dispatch("idx_pool", pl, b1 - b0);
            gpu_.barrier();
        }
        if (b1 <= TOPB) return 0;
        PcIdxScore ps{iq_.addr(), kb, scores_.addr(), (uint32_t)pos_, (uint32_t)R, (uint32_t)c_.idx_heads, (uint32_t)nblk_};
        gpu_.dispatch("idx_score", ps, groups(b1, 256), n);
        gpu_.barrier();
        PcIdxSel pt{scores_.addr(), isel_.addr(), (uint32_t)pos_, (uint32_t)R, (uint32_t)TOPB, (uint32_t)nblk_};
        gpu_.dispatch("idx_select", pt, n);
        gpu_.barrier();
        return isel_.addr();
    }

    void norm(Ptr x, uint64_t delta, Ptr w, Ptr out, int n) {
        PcNorm pc{x.addr(), delta, w.addr(), out.addr(), (uint32_t)c_.hidden, c_.rms_eps};
        gpu_.dispatch("add_rmsnorm", pc, n);
    }

    void deltanet(int l, int n) {
        const GLayer& g = L_[l];
        const int s = slot_[l];
        lin(g.qkv, xn_, qkv_, n);
        lin(g.z, xn_, z_, n);
        mat_b(g.a, xn_.addr(), a_.addr(), n, c_.hidden);
        mat_b(g.b, xn_.addr(), b_.addr(), n, c_.hidden);
        gpu_.barrier();
        const VkDeviceSize cs = (VkDeviceSize)(c_.conv_kernel - 1) * c_.conv_dim() * 4;
        PcConv pcc{qkv_.addr(), g.conv.addr(), conv_state_->addr + s * cs, (uint32_t)n, (uint32_t)c_.conv_dim()};
        gpu_.dispatch("conv1d", pcc, groups(c_.conv_dim(), 256));
        gpu_.barrier();
        const VkDeviceSize rs = (VkDeviceSize)c_.lin_v_heads * c_.lin_k_dim * c_.lin_v_dim * 4;
        PcDelta pcd{qkv_.addr(), a_.addr(), b_.addr(), g.neg_a.addr(), g.dt_bias.addr(), rec_state_->addr + s * rs,
                    core_.addr(), (uint32_t)n, (uint32_t)c_.conv_dim(), (uint32_t)c_.lin_key_dim(),
                    (uint32_t)c_.lin_v_heads, (uint32_t)(c_.lin_v_heads / c_.lin_k_heads)};
        gpu_.dispatch("deltanet", pcd, c_.lin_v_heads, c_.lin_v_dim / 16);
        gpu_.barrier();
        PcGatedNorm pcg{core_.addr(), z_.addr(), g.lnorm.addr(), (uint32_t)c_.lin_v_heads, c_.rms_eps, c_.qwen4 ? 1u : 0u};
        gpu_.dispatch("gated_norm", pcg, n, c_.lin_v_heads);
        gpu_.barrier();
        lin(g.out, core_, mix_, n);
    }

    void attention(int l, int n) {
        const GLayer& g = L_[l];
        const VkDeviceSize kv = (VkDeviceSize)slot_[l] * max_ctx_ * c_.n_kv_heads * c_.head_dim * 4;
        lin(g.q, xn_, qg_, n);
        lin(g.k, xn_, kb_, n);
        lin(g.v, xn_, vb_, n);
        gpu_.barrier();
        const uint64_t sel = c_.qwen4 ? indexer(l, n) : 0;
        PcQkRope pr{qg_.addr(), kb_.addr(), vb_.addr(), qo_.addr(), kcache_->addr + kv, vcache_->addr + kv,
                    g.qn.addr(), g.kn.addr(), rope_.addr(), (uint32_t)pos_, (uint32_t)c_.n_heads,
                    (uint32_t)c_.n_kv_heads, (uint32_t)c_.rope_dim, c_.rms_eps};
        gpu_.dispatch("qk_norm_rope", pr, n, c_.n_heads + c_.n_kv_heads);
        gpu_.barrier();
        // split-K over keys: many splits when few tokens (decode), few when the batch is already wide
        const int maxlen = pos_ + n;
        int splits = n <= NT ? (maxlen + 255) / 256 : (maxlen + 1023) / 1024;
        splits = std::max(1, std::min(splits, std::min(64, ATT_PART_MAX / n)));
        const int chunk = (maxlen + splits - 1) / splits;
        PcAttn pa{qo_.addr(), kcache_->addr + kv, vcache_->addr + kv, part_.addr(), sel, (uint32_t)pos_,
                  (uint32_t)c_.n_heads, (uint32_t)c_.n_kv_heads, (uint32_t)chunk, (uint32_t)splits,
                  1.0f / std::sqrt((float)c_.head_dim), (uint32_t)c_.idx_ratio,
                  (uint32_t)(c_.idx_ratio ? c_.idx_budget / c_.idx_ratio : 0)};
        gpu_.dispatch("attention", pa, n, c_.n_heads, splits);
        gpu_.barrier();
        PcAttnCombine pc{part_.addr(), qg_.addr(), att_.addr(), (uint32_t)c_.n_heads, (uint32_t)splits};
        gpu_.dispatch("attn_combine", pc, n, c_.n_heads);
        gpu_.barrier();
        lin(g.o, att_, mix_, n);
    }

    void moe(int l, int n) {
        const GLayer& g = L_[l];
        const int H = c_.hidden, TK = c_.top_k, E = c_.n_experts, I = c_.moe_inter;
        const uint64_t sel = sel_.addr() + (uint64_t)l * NMAX * TK * 4;
        mat_b(g.router, xn_.addr(), rlogits_.addr(), n, H, "router");
        mat_b(g.sgate, xn_.addr(), sgate_.addr(), n, H, "shared_gate");
        if (g.sh_gate.q.rows) {
            mat_swiglu(g.sh_gate.q, g.sh_up.q, xn_, sact_, n);
            gpu_.barrier();
        } else {
            mat_b(g.sh_gate.b, xn_.addr(), sact_.addr(), n, H);
            mat_b(g.sh_up.b, xn_.addr(), su_.addr(), n, H);
            gpu_.barrier();
            ew(sact_, su_, (uint32_t)(n * c_.shared_inter), 1, 1.0f);
            gpu_.barrier();
        }
        lin(g.sh_down, sact_, sy_, n);
        PcRoute pr{rlogits_.addr(), sel, selw_.addr(), (uint32_t)E, (uint32_t)TK};
        gpu_.dispatch("moe_route", pr, n);
        gpu_.barrier();
        if (hybrid_) {
            moe_hybrid(l, n);
            return;
        }

        if (!pinned_) {
            // Fallback: experts must be resident before they run, so stop and upload the misses now.
            const VkDeviceSize so = (VkDeviceSize)l * NMAX * TK * 4;
            gpu_.copy(sel_.b, sel_.off + so, hsel_, so, (VkDeviceSize)n * TK * 4);
            gpu_.barrier();
            gpu_.submit_wait();
            tick_++;
            const uint32_t* s = (const uint32_t*)(hsel_->map + (size_t)l * NMAX * TK * 4);
            gpu_.begin();
            for (int i = 0; i < n * TK; i++) {
                const int key = l * E + (int)s[i];
                usage_[key]++;
                if (where_[key] >= 0) { hits_++; slots_[where_[key]].tick = tick_; continue; }
                misses_++;
                if (!fill_slot(key)) throw std::runtime_error("expert cache too small for this batch");
            }
            up_.flush();  // staged expert data (separate submission, before the loc updates execute)
            gpu_.barrier();
        }

        const bool gemm = n >= EXPERT_GEMM_MIN;
        const uint32_t chunk = gemm ? 64 : NT;
        const uint32_t rpw = gemm ? 64 : 8;  // rows per workgroup
        PcGroup pg{sel, selw_.addr(), loc_.addr() + (uint64_t)l * E * 8, entries_.addr(), toks_.addr(), comb_.addr(),
                   args_.addr(), (uint32_t)(n * TK), (uint32_t)TK, (uint32_t)E, chunk, groups(I, rpw), groups(H, rpw)};
        gpu_.dispatch("moe_group", pg, 1);
        gpu_.barrier();
        PcExpert pu{entries_.addr(), toks_.addr(), xn_.addr(), act_.addr(), 0, (uint32_t)off_sg_, (uint32_t)off_up_,
                    (uint32_t)off_su_, (uint32_t)I, (uint32_t)H, (uint32_t)H, (uint32_t)I, 0};
        const bool tc = gpu_.coop();
        gpu_.dispatch_indirect(gemm ? (tc ? "gemm_coop_expert_swiglu" : "gemm_fp8_expert_swiglu") : "matmul_fp8_expert_swiglu", pu, args_);
        gpu_.barrier();
        PcExpert pd{entries_.addr(), toks_.addr(), act_.addr(), ey_.addr(), (uint32_t)off_down_, (uint32_t)off_sd_, 0, 0,
                    (uint32_t)H, (uint32_t)I, (uint32_t)I, (uint32_t)H, 1};
        gpu_.dispatch_indirect(gemm ? (tc ? "gemm_coop_expert" : "gemm_fp8_expert") : "matmul_fp8_expert", pd, {args_.b, args_.off + 12});
        gpu_.barrier();
        // qwen4 writes the block output (mixed into the streams by hc_out); otherwise add to the residual
        PcCombine pc{c_.qwen4 ? mix_.addr() : x_.addr(), sgate_.addr(), sy_.addr(), ey_.addr(), comb_.addr(), (uint32_t)H,
                     (uint32_t)TK, c_.qwen4 ? 0u : 1u};
        gpu_.dispatch("moe_combine", pc, groups(H, 256), n);
        gpu_.barrier();
    }

    // Hybrid MoE for layer l (command buffer open, routing recorded). Leaves a command buffer open.
    void moe_hybrid(int l, int n) {
        const int H = c_.hidden, TK = c_.top_k, E = c_.n_experts, I = c_.moe_inter;
        const VkDeviceSize so = (VkDeviceSize)l * NMAX * TK * 4;
        gpu_.copy(sel_.b, sel_.off + so, hsel_, so, (VkDeviceSize)n * TK * 4);
        gpu_.copy(xn_.b, xn_.off, hxn_, 0, (VkDeviceSize)n * H * 4);
        gpu_.barrier();
        gpu_.submit_wait();

        // CPU: split selections into cached (GPU) and missed (CPU) experts, build the GPU tables
        const uint32_t* sel = (const uint32_t*)(hsel_->map + so);
        std::vector<std::vector<int>> per(E);  // expert -> selection indices (t * TK + k)
        for (int i = 0; i < n * TK; i++) per[sel[i]].push_back(i);
        const bool gemm = n >= EXPERT_GEMM_MIN;
        const uint32_t chunk = gemm ? 64 : NT;
        ExpertEntry* ent = (ExpertEntry*)htab_->map;
        uint32_t* toks = (uint32_t*)(htab_->map + htab_toks_);
        uint32_t* comb = (uint32_t*)(htab_->map + htab_comb_);
        uint32_t ne = 0, row = 0;
        std::vector<int> cpu_experts;
        for (int e = 0; e < E; e++) {
            if (per[e].empty()) continue;
            const int key = l * E + e;
            usage_[key]++;
            if (where_[key] < 0) {
                misses_ += per[e].size();
                cpu_experts.push_back(e);
                if ((int)pending_.size() < MAX_UPLOADS_HYBRID && std::find(pending_.begin(), pending_.end(), key) == pending_.end())
                    pending_.push_back(key);
                continue;
            }
            hits_ += per[e].size();
            slots_[where_[key]].tick = tick_;
            const uint64_t base = slot_addr(where_[key]);
            for (size_t i0 = 0; i0 < per[e].size(); i0 += chunk) {
                const uint32_t cnt = (uint32_t)std::min<size_t>(chunk, per[e].size() - i0);
                ent[ne++] = {base, row, cnt};
                for (uint32_t j = 0; j < cnt; j++) {
                    const int i = per[e][i0 + j];
                    toks[row + j] = (uint32_t)(i / TK);
                    comb[2 * i] = row + j;
                }
                row += cnt;
            }
        }
        const uint32_t gpu_rows = row;

        // GPU: cached experts (runs while the CPU computes the misses)
        gpu_.begin();
        if (ne) {
            const uint64_t tab = htab_->addr, tk = htab_->addr + htab_toks_;
            PcExpert pu{tab, tk, xn_.addr(), act_.addr(), 0, (uint32_t)off_sg_, (uint32_t)off_up_, (uint32_t)off_su_,
                        (uint32_t)I, (uint32_t)H, (uint32_t)H, (uint32_t)I, 0};
            const bool tc = gpu_.coop();
            gpu_.dispatch(gemm ? (tc ? "gemm_coop_expert_swiglu" : "gemm_fp8_expert_swiglu") : "matmul_fp8_expert_swiglu", pu,
                          groups(I, gemm ? 64 : 8), ne);
            gpu_.barrier();
            PcExpert pd{tab, tk, act_.addr(), ey_.addr(), (uint32_t)off_down_, (uint32_t)off_sd_, 0, 0, (uint32_t)H,
                        (uint32_t)I, (uint32_t)I, (uint32_t)H, 1};
            gpu_.dispatch(gemm ? (tc ? "gemm_coop_expert" : "gemm_fp8_expert") : "matmul_fp8_expert", pd, groups(H, gemm ? 64 : 8), ne);
        }
        gpu_.submit();

        // CPU: missed experts, straight from the mmapped weights
        const float* xn = (const float*)hxn_->map;
        float* cy = (float*)hcy_->map;
        size_t rows_cpu = 0;
        for (int e : cpu_experts) rows_cpu += per[e].size();
        if (rows_cpu) {
            std::vector<float> xin(rows_cpu * H), g(rows_cpu * I), u(rows_cpu * I);
            std::vector<ck::MatJob> jobs;
            size_t r = 0;
            for (int e : cpu_experts) {
                const Expert& ex = w_.layers[l].experts[e];
                for (size_t j = 0; j < per[e].size(); j++) {
                    const int i = per[e][j];
                    std::memcpy(&xin[(r + j) * H], xn + (size_t)(i / TK) * H, (size_t)H * 4);
                    comb[2 * i] = gpu_rows + (uint32_t)(r + j);
                }
                const int cnt = (int)per[e].size();
                jobs.push_back({&ex.gate, nullptr, &xin[r * H], &g[r * I], cnt});
                jobs.push_back({&ex.up, nullptr, &xin[r * H], &u[r * I], cnt});
                r += cnt;
            }
            ck::run_jobs(*cpu_, jobs);
            cpu_->parallel_for((int64_t)g.size(), [&](int64_t b, int64_t e2) {
                for (int64_t i = b; i < e2; i++) g[i] = ck::silu(g[i]) * u[i];
            }, 4096);
            jobs.clear();
            r = 0;
            for (int e : cpu_experts) {
                const int cnt = (int)per[e].size();
                jobs.push_back({&w_.layers[l].experts[e].down, nullptr, &g[r * I], cy + r * H, cnt});
                r += cnt;
            }
            ck::run_jobs(*cpu_, jobs);
            cpu_rows_ += rows_cpu;
        }
        gpu_.wait();

        // GPU: bring CPU rows next to the GPU rows, then combine (weights from moe_route via selw_)
        gpu_.begin();
        if (rows_cpu) {
            gpu_.copy(hcy_, 0, ey_.b, ey_.off + (VkDeviceSize)gpu_rows * H * 4, (VkDeviceSize)rows_cpu * H * 4);
        }
        // comb rows come from the host table; weights are filled on the GPU from selw_
        gpu_.copy(htab_, htab_comb_, comb_.b, comb_.off, (VkDeviceSize)n * TK * 8);
        gpu_.barrier();
        PcCombW pw{comb_.addr(), selw_.addr(), (uint32_t)(n * TK)};
        gpu_.dispatch("comb_weights", pw, groups(n * TK, 256));
        gpu_.barrier();
        // qwen4 writes the block output (mixed into the streams by hc_out); otherwise add to the residual
        PcCombine pc{c_.qwen4 ? mix_.addr() : x_.addr(), sgate_.addr(), sy_.addr(), ey_.addr(), comb_.addr(), (uint32_t)H,
                     (uint32_t)TK, c_.qwen4 ? 0u : 1u};
        gpu_.dispatch("moe_combine", pc, groups(H, 256), n);
        gpu_.barrier();
    }

    void forward_chunk(const int* tokens, int n, float* logits) {
        const int H = c_.hidden, S = c_.hc;
        float* hx = (float*)hx_->map;
        for (int t = 0; t < n; t++) {  // qwen4: every stream starts as the embedding
            const uint16_t* e = w_.embed.w + (size_t)tokens[t] * H;
            for (int i = 0; i < H; i++) hx[(size_t)t * HH_ + i] = bf16_to_f32(e[i]);
            for (int s = 1; s < S; s++) std::memcpy(&hx[(size_t)t * HH_ + s * H], &hx[(size_t)t * HH_], H * 4);
        }
        if (hemb_) {
            ck::ple_embed(*ple_pool_, c_, w_.ple, hist_, tokens, n, (float*)hemb_->map);
            const int C = (int)hist_.size();
            std::vector<int> all(hist_);
            all.insert(all.end(), tokens, tokens + n);
            for (int i = 0; i < C; i++) hist_[i] = all[all.size() - C + i];
        }
        tick_++;
        gpu_.begin();
        // cache fills queued by the previous pass
        for (int key : pending_)
            if (where_[key] < 0 && !fill_slot(key)) break;
        up_.flush();  // non-pinned fills go through the staging buffer
        pending_.clear();
        gpu_.copy(hx_, 0, x_.b, x_.off, (VkDeviceSize)n * HH_ * 4);
        if (hemb_) gpu_.copy(hemb_, 0, emb_.b, emb_.off, (VkDeviceSize)n * c_.ple_dim * 4);
        gpu_.barrier();
        for (int l = 0; l < c_.n_layers; l++) {
            const GLayer& g = L_[l];
            if (c_.qwen4) {
                if (l == c_.ple_layer) ple(n);
                hc_in(g.attn_hc, x_, n);
                if (w_.layers[l].full_attn) attention(l, n);
                else deltanet(l, n);
                gpu_.barrier();
                hc_out(n);
                hc_in(g.mlp_hc, x_, n);
                moe(l, n);
                hc_out(n);
                continue;
            }
            norm(x_, 0, g.in_norm, xn_, n);
            gpu_.barrier();
            if (w_.layers[l].full_attn) attention(l, n);
            else deltanet(l, n);
            gpu_.barrier();
            norm(x_, mix_.addr(), g.post_norm, xn_, n);
            gpu_.barrier();
            moe(l, n);
        }
        if (logits) {
            const Ptr xl{x_.b, x_.off + (VkDeviceSize)(n - 1) * HH_ * 4};
            if (c_.qwen4) hc_in(mixer_, xl, 1);
            else norm(xl, 0, final_norm_, xn_, 1);
            gpu_.barrier();
            mat_b(lm_head_, xn_.addr(), logits_.addr(), 1, H, "lm_head");
            gpu_.barrier();
            gpu_.copy(logits_.b, logits_.off, hlog_, 0, (VkDeviceSize)c_.vocab * 4);
        }
        if (pinned_)  // selected expert ids of all layers, for cache bookkeeping
            for (int l = 0; l < c_.n_layers; l++) {
                const VkDeviceSize so = (VkDeviceSize)l * NMAX * c_.top_k * 4;
                gpu_.copy(sel_.b, sel_.off + so, hsel_, so, (VkDeviceSize)n * c_.top_k * 4);
            }
        gpu_.barrier();
        gpu_.submit_wait();
        if (logits) std::memcpy(logits, hlog_->map, (size_t)c_.vocab * 4);
        if (pinned_)
            for (int l = 0; l < c_.n_layers; l++) account(l, n);
        pos_ += n;
    }
};

}  // namespace

std::unique_ptr<Backend> make_vulkan_backend(const Config& cfg, const ModelWeights& w, int max_ctx, const VulkanOptions& opt) {
    return std::make_unique<VulkanBackend>(cfg, w, max_ctx, opt);
}
