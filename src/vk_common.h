// Vulkan device wrapper shared by the compute backends: buffers addressed by device address, command
// recording helpers, a device-memory arena and a staging uploader.
#pragma once
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "shaders_spv.h"

#define VK_CHECK(x)                                                                                     \
    do {                                                                                                \
        VkResult r_ = (x);                                                                              \
        if (r_ != VK_SUCCESS) throw std::runtime_error("Vulkan error " + std::to_string(r_) + ": " #x); \
    } while (0)

namespace vkc {

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
        VK_CHECK(vkAllocateCommandBuffers(dev_, &cai, &cmd2_));
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
    // Submits the recorded commands without waiting and starts recording the ones that run after them (on a
    // second command buffer); the next submit_wait() covers both. The CPU can work while the first part runs.
    void submit_chain() {
        if (profile_) {  // the timestamp bookkeeping is per submission
            submit_wait();
            begin();
            return;
        }
        VK_CHECK(vkEndCommandBuffer(cmd_));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd_;
        VK_CHECK(vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE));
        std::swap(cmd_, cmd2_);
        begin();
    }
    void wait() {
        // Short submissions (one per layer in the hybrid MoE path) finish in well under a millisecond: poll
        // before falling back to a blocking wait, whose wake-up alone can cost that much.
        for (int i = 0; i < 20000 && vkGetFenceStatus(dev_, fence_) == VK_NOT_READY; i++) {}
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
    VkCommandBuffer cmd2_ = VK_NULL_HANDLE;
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

}  // namespace vkc
