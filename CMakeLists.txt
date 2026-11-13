cmake_minimum_required(VERSION 3.20)
project(qwen_local CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(QW_VULKAN "Build the Vulkan compute backend" OFF)
option(QW_NATIVE "Optimize for the build machine (AVX-512 etc.)" ON)

add_executable(qwen
    src/main.cpp
    src/model.cpp
    src/safetensors.cpp
    src/tokenizer.cpp
    src/cpu_backend.cpp
    src/cpu_kernels.cpp
    src/gguf.cpp
    src/quant.cpp
    src/qwen4.cpp
    src/qwen4_cpu.cpp)

if(MSVC)
    target_compile_options(qwen PRIVATE /O2 /utf-8 /W3 /bigobj /fp:fast)
    if(QW_NATIVE)
        target_compile_options(qwen PRIVATE /arch:AVX512)
    endif()
else()
    target_compile_options(qwen PRIVATE -O3 -Wall -ffast-math)
    if(QW_NATIVE)
        target_compile_options(qwen PRIVATE -march=native)
    endif()
    find_package(Threads REQUIRED)
    target_link_libraries(qwen PRIVATE Threads::Threads)
endif()

if(QW_VULKAN)
    find_package(Vulkan REQUIRED COMPONENTS glslc)
    # name source defines
    set(QW_SHADERS
        "matmul_fp8|matmul_fp8.comp|"
        "matmul_fp8_swiglu|matmul_fp8.comp|-DSWIGLU"
        "matmul_fp8_expert|matmul_fp8.comp|-DEXPERT"
        "matmul_fp8_expert_swiglu|matmul_fp8.comp|-DEXPERT -DSWIGLU"
        "matmul_bf16|matmul_bf16.comp|"
        "add_rmsnorm|add_rmsnorm.comp|"
        "conv1d|conv1d.comp|"
        "deltanet|deltanet.comp|"
        "gated_norm|gated_norm.comp|"
        "qk_norm_rope|qk_norm_rope.comp|"
        "attention|attention.comp|"
        "moe_combine|moe_combine.comp|"
        "moe_route|moe_route.comp|"
        "moe_group|moe_group.comp|"
        "attn_combine|attn_combine.comp|"
        "comb_weights|comb_weights.comp|"
        "gemm_fp8|gemm_fp8.comp|"
        "gemm_fp8_swiglu|gemm_fp8.comp|-DSWIGLU"
        "gemm_fp8_expert|gemm_fp8.comp|-DEXPERT"
        "gemm_fp8_expert_swiglu|gemm_fp8.comp|-DEXPERT -DSWIGLU"
        "gemm_coop|gemm_coop.comp|"
        "gemm_coop_swiglu|gemm_coop.comp|-DSWIGLU"
        "gemm_coop_expert|gemm_coop.comp|-DEXPERT"
        "gemm_coop_expert_swiglu|gemm_coop.comp|-DEXPERT -DSWIGLU"
        # qwen4exp (GGUF quantization codecs, hyper-connections, PLE, QSA)
        "q4_matmul_q8|q4_matmul.comp|-DC_Q8"
        "q4_matmul_q8_swiglu|q4_matmul.comp|-DC_Q8 -DSWIGLU"
        "q4_matmul_f32|q4_matmul.comp|-DC_F32"
        "q4_matmul_f32_swiglu|q4_matmul.comp|-DC_F32 -DSWIGLU"
        "q4_expert_q4k_swiglu|q4_matmul.comp|-DC_Q4K -DEXPERT -DSWIGLU"
        "q4_expert_q5k_swiglu|q4_matmul.comp|-DC_Q5K -DEXPERT -DSWIGLU"
        "q4_expert_q8_swiglu|q4_matmul.comp|-DC_Q8 -DEXPERT -DSWIGLU"
        "q4_expert_q51|q4_matmul.comp|-DC_Q51 -DEXPERT"
        "q4_expert_q8|q4_matmul.comp|-DC_Q8 -DEXPERT"
        "q4_expert_q4k|q4_matmul.comp|-DC_Q4K -DEXPERT"
        "q4_expert_q5k|q4_matmul.comp|-DC_Q5K -DEXPERT"
        "q4_gemm_q8|q4_gemm.comp|-DC_Q8"
        "q4_gemm_q8_swiglu|q4_gemm.comp|-DC_Q8 -DSWIGLU"
        "q4_gemm_f32|q4_gemm.comp|-DC_F32"
        "q4_gemm_f32_swiglu|q4_gemm.comp|-DC_F32 -DSWIGLU"
        "q4_gemm_expert_q4k_swiglu|q4_gemm.comp|-DC_Q4K -DEXPERT -DSWIGLU"
        "q4_gemm_expert_q5k_swiglu|q4_gemm.comp|-DC_Q5K -DEXPERT -DSWIGLU"
        "q4_gemm_expert_q8_swiglu|q4_gemm.comp|-DC_Q8 -DEXPERT -DSWIGLU"
        "q4_gemm_expert_q51|q4_gemm.comp|-DC_Q51 -DEXPERT"
        "q4_gemm_expert_q8|q4_gemm.comp|-DC_Q8 -DEXPERT"
        "q4_gemm_expert_q4k|q4_gemm.comp|-DC_Q4K -DEXPERT"
        "q4_gemm_expert_q5k|q4_gemm.comp|-DC_Q5K -DEXPERT"
        "q4_hc_norm|q4_hc.comp|-DHC_NORM"
        "q4_hc_mix|q4_hc.comp|-DHC_MIX"
        "q4_hc_combine|q4_hc.comp|-DHC_COMBINE"
        "q4_ple_gate|q4_ple.comp|-DPLE_GATE"
        "q4_ple_conv|q4_ple.comp|-DPLE_CONV"
        "gated_norm_sigmoid|gated_norm.comp|-DGATE_SIGMOID"
        "deltanet_tiled|deltanet.comp|-DV_TILED"
        "deltanet_tiled_snap|deltanet.comp|-DV_TILED -DSNAP"
        "conv1d_snap|conv1d.comp|-DSNAP"
        "q4_hc_cat|q4_hc.comp|-DHC_CAT"
        "moe_combine_set|moe_combine.comp|-DASSIGN"
        "qk_norm_rope_kv16|qk_norm_rope.comp|-DKV16"
        "attention_kv16|attention.comp|-DKV16"
        "attention_qsa_kv16|attention.comp|-DQSA -DKV16")
    set(spv_files "")
    set(spv_args "")
    file(GLOB shader_deps ${CMAKE_SOURCE_DIR}/shaders/*)
    foreach(s ${QW_SHADERS})
        string(REPLACE "|" ";" parts "${s}")
        list(GET parts 0 name)
        list(GET parts 1 src)
        list(GET parts 2 defs)
        separate_arguments(defs)
        set(out ${CMAKE_BINARY_DIR}/spv/${name}.spv)
        add_custom_command(OUTPUT ${out}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/spv
            COMMAND Vulkan::glslc --target-env=vulkan1.3 -O ${defs} ${CMAKE_SOURCE_DIR}/shaders/${src} -o ${out}
            DEPENDS ${shader_deps} VERBATIM)
        list(APPEND spv_files ${out})
        list(APPEND spv_args "${name}=${out}")
    endforeach()
    list(JOIN spv_args "\n" spv_lines)
    file(WRITE ${CMAKE_BINARY_DIR}/spv_list.txt "${spv_lines}\n")
    add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/gen/shaders_spv.h
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/gen
        COMMAND ${CMAKE_COMMAND} -DOUT=${CMAKE_BINARY_DIR}/gen/shaders_spv.h -DLIST=${CMAKE_BINARY_DIR}/spv_list.txt -P ${CMAKE_SOURCE_DIR}/cmake/embed_spv.cmake
        DEPENDS ${spv_files} ${CMAKE_SOURCE_DIR}/cmake/embed_spv.cmake VERBATIM)
    target_sources(qwen PRIVATE src/vk_backend.cpp src/qwen4_vk.cpp ${CMAKE_BINARY_DIR}/gen/shaders_spv.h)
    target_include_directories(qwen PRIVATE ${CMAKE_BINARY_DIR}/gen)
    target_compile_definitions(qwen PRIVATE QW_VULKAN=1)
    target_link_libraries(qwen PRIVATE Vulkan::Vulkan)
endif()
