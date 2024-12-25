#include <iostream>
#include <time.h>
#include <map>
#include <vector>
#include <string>
#include <algorithm>
#include <sstream>
#include <iterator>
#include <iostream>
#include <thread>
#include "include/cxxopts.hpp"
#include <cuda_runtime.h>


#define SDC_MEMCPY_VERSION "v0.1.0"


// Device code
__global__ void MemXOR(int* src, int* dest, unsigned long long size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        dest[idx] = src[idx] ^ 0xFFFFFFFF;
    }
}

auto get_gpus(cxxopts::ParseResult& result) -> std::vector<int>
{
    auto& gpus = result["gpus"].as<std::vector<std::string>>();

    std::vector<int> gpu_ids;

    int gpus_count;
    cudaError_t error = cudaGetDeviceCount(&gpus_count);

    if (error != cudaSuccess) {
        std::cerr << "Error: " << cudaGetErrorString(error) << std::endl;
        return gpu_ids;
    }

    if (gpus[0] == "all")
    {
        for (int i = 0; i < gpus_count; i++)
        {
            gpu_ids.push_back(i);
        }
        return gpu_ids;
    }

    for (auto& gpu : gpus)
    {
        if (gpu == "all")
        {
            continue;
        }
        else
        {
            int gpu_id = std::stoi(gpu);
            if (gpu_id >= gpus_count)
            {
                std::cerr << "Error: GPU " << gpu_id << " does not exist" << std::endl;
                continue;
            }
            gpu_ids.push_back(gpu_id);
        }
    }

    return gpu_ids;
}

auto get_duration(cxxopts::ParseResult& result) -> unsigned long long
{
    auto s_duration = result["duration"].as<std::string>();
    auto unit = s_duration[s_duration.size() - 1];
    auto time = 100;
    if (s_duration.find(unit) != std::string::npos)
    {
        time = std::stoi(s_duration.substr(0, s_duration.size() - 1));
    }
    auto time_multiplier = 1;
    switch (unit)
    {
    case 's':
    case 'S':
        time_multiplier = 1;
        break;
    case 'm':
    case 'M':
        time_multiplier = 60;
        break;
    case 'h':
    case 'H':
        time_multiplier = 60 * 60;
        break;
    case 'd':
    case 'D':
        time_multiplier = 60 * 60 * 24;
        break;
    default:
        break;
    }
    unsigned long long duration = time * time_multiplier;
    return duration;
}

auto get_size(cxxopts::ParseResult& result) -> unsigned long long
{
    auto s_size = result["size"].as<std::string>();
    auto unit = s_size[s_size.size() - 1];
    if (s_size.find(unit) != std::string::npos)
    {
        auto size = std::stoi(s_size.substr(0, s_size.size() - 1));
    }
    auto size_multiplier = 1;
    switch (unit)
    {
        case 'b':
        case 'B':
            size_multiplier = 1;
            break;
        case 'k':
        case 'K':
            size_multiplier = 1024;
            break;
        case 'm':
        case 'M':
            size_multiplier = 1024 * 1024;
            break;
        case 'g':
        case 'G':
            size_multiplier = 1024 * 1024 * 1024;
            break;
        default:
            break;
    }
    unsigned long long size = std::stoi(s_size) * size_multiplier;
    return size;
}

auto parse_args(cxxopts::ParseResult&result) -> int
{
    auto gpus = result["gpus"].as<std::vector<std::string>>();
    if (gpus.size() == 0)
    {
        std::cerr << "No GPUs specified" << std::endl;
        return 1;
    }
    for (auto& gpu : gpus)
    {
        if (gpu == "all")
        {
            gpu = "0";
        }
    }

    return 0;
}

auto print_help(cxxopts::ParseResult& result, auto& options) -> int
{
    std::cout << options.help() << std::endl;
    return 0;
}

auto print_version(cxxopts::ParseResult& result, auto& options) -> int
{
    std::cout << SDC_MEMCPY_VERSION << std::endl;
    return 0;
}

auto list_gpus(cxxopts::ParseResult& result, auto& options) -> int
{
    std::cout << "List of GPUs" << std::endl;
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess) {
        std::cerr << "Error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }

    std::cout << "Number of CUDA devices: " << deviceCount << std::endl;
    std::cout << "==============================================================" << std::endl;
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        std::cout << "Device " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "  Registers per Block: " << prop.regsPerBlock << std::endl;
        std::cout << "  Warp Size: " << prop.warpSize << std::endl;
        std::cout << "  Memory Clock Rate: " << prop.memoryClockRate / (1000.0) << " MHz" << std::endl;
        std::cout << "  Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
        std::cout << "  Peak Memory Bandwidth: " << 2.0 * prop.memoryClockRate * prop.memoryBusWidth / 8 / 1024 / 1024 << " GB/s" << std::endl;
        std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Max Threads per MultiProcessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "  Max Blocks per MultiProcessor: " << prop.maxBlocksPerMultiProcessor << std::endl;
        std::cout << "  Max Grid Size: " << prop.maxGridSize[0] << " " << prop.maxGridSize[1] << " " << prop.maxGridSize[2] << std::endl;
        std::cout << "  Max Threads Dimension: " << prop.maxThreadsDim[0] << " " << prop.maxThreadsDim[1] << " " << prop.maxThreadsDim[2] << std::endl;
        std::cout << "  Concurrent Kernels: " << prop.concurrentKernels << std::endl;
        std::cout << "  Concurrent Memory Accesses: " << prop.concurrentManagedAccess << std::endl;
        std::cout << "  ECC Enabled: " << (prop.ECCEnabled ? "Yes" : "No") << std::endl;
        std::cout << "  Unified Addressing: " << (prop.unifiedAddressing ? "Yes" : "No") << std::endl;
        std::cout << "  Async Engine Count: " << prop.asyncEngineCount << std::endl;
        std::cout << "  Device Overlap: " << (prop.deviceOverlap ? "Yes" : "No") << std::endl;
        std::cout << "  Kernel Execution Timeout: " << (prop.kernelExecTimeoutEnabled ? "Yes" : "No") << std::endl;
        std::cout << "  Can Map Host Memory: " << (prop.canMapHostMemory ? "Yes" : "No") << std::endl;
        std::cout << "  Compute Mode: " << (prop.computeMode == cudaComputeModeDefault ? "Default" : "Exclusive") << std::endl;
        std::cout << "  Shared Memory per Multiprocessor: " << prop.sharedMemPerMultiprocessor / 1024 << " KB" << std::endl;
        std::cout << "  L2 Cache Size: " << prop.l2CacheSize / 1024 << " KB" << std::endl;
        std::cout << "  Shared Memory per Multiprocessor: " << prop.sharedMemPerMultiprocessor / 1024 << " KB" << std::endl;
        std::cout << "  L2 Cache Size: " << prop.l2CacheSize / 1024 << " KB" << std::endl;
        std::cout << "--------------------------------------------------------------" << std::endl;
    }

    return 0;
}


void load_alternating(void* a, void* b, size_t size)
{
    char* a_ptr = (char*)a;
    for (size_t i = 0; i < size; i++) {
        a_ptr[i] = 0b10101010;
    }
}

bool validate_alternating(void* a, void* b, size_t size)
{
    char* a_ptr = (char*)a;
    char* b_ptr = (char*)b;
    for (size_t i = 0; i < size; i++) {
        if (a_ptr[i] != 0 && b_ptr[i]!= 0 && a_ptr[i] & b_ptr[i] != 0) {
            return false;
        }
    }
    return true;
}

auto load_pattern_map() -> std::map<std::string, std::function<void(void*, void*, size_t)>>
{
    std::map<std::string, std::function<void(void*, void*, size_t)>> patterns;
    patterns["alternating"] = load_alternating;
    return patterns;
}

auto load_validate_map() -> std::map<std::string, std::function<bool(void*, void*, size_t)>>
{
    std::map<std::string, std::function<bool(void*, void*, size_t)>> patterns;
    patterns["alternating"] = validate_alternating;
    return patterns;
}


auto memcpy_thread_fxn(int gpu_id, bool pinned, bool unified, bool zero_buffer, bool non_blocking, unsigned long long size, int repeat, unsigned long long duration, std::vector<std::string> patterns) -> void
{
    cudaSetDevice(gpu_id);

    // Load functions.
    auto pattern_map = load_pattern_map();
    auto validate_map = load_validate_map();

    std::cout << "Allocating Host Memory" << std::endl;
    void* h_a;
    void* h_b;

    if (pinned)
    {
        std::cout << "Allocating Pinned Memory" << std::endl;
        cudaMallocHost(&h_a, size);
        cudaMallocHost(&h_b, size);
    }
    else
    {
        cudaHostAlloc(&h_a, size, cudaHostAllocDefault);
        cudaHostAlloc(&h_b, size, cudaHostAllocDefault);
    }

    std::cout << "Allocating Device Memory" << std::endl;
    void* d_a;
    void* d_b;
    if (unified)
    {
        std::cout << "Allocating Unified Memory" << std::endl;
        cudaMallocManaged(&d_a, size);
        cudaMallocManaged(&d_b, size);  
    }
    else
    {
        cudaMalloc(&d_a, size);
        cudaMalloc(&d_b, size);
    }

    if (zero_buffer)
    {
        std::cout << "Zeroing Device Memory" << std::endl;
        cudaMemset(d_a, 0, size);
        cudaMemset(d_b, 0, size);
        memset(h_a, 0, size);
        memset(h_b, 0, size);
    }

    for (int i = 0; i < repeat; i++)
    {
        std::cout << "Running Repeat: " << i << " on GPU " << gpu_id << std::endl;
        for (auto pattern : patterns)
        {
            std::cout << "Running Pattern: " << pattern << std::endl;
            auto pattern_func = pattern_map[pattern];
            auto validate_func = validate_map[pattern];
            pattern_func(h_a, h_b, size);
            auto start = time(NULL);
            while (time(NULL) - start < duration)
            {
                if(zero_buffer)
                {
                    cudaMemset(d_a, 0, size);
                    cudaMemset(d_b, 0, size);
                    memset(h_b, 0, size);
                }
                cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
                MemXOR<<<1024, 1>>>((int *)d_a, (int *)d_b, size / 4);
                cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                if (!validate_func(h_a, h_b, size))
                {
                    std::cout << " [!] Found corruption on GPU " << gpu_id << "" << std::endl;
                }
            }
        }
    }


    std::cout << "Freeing Memory" << std::endl;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
}

auto run_memcpy(cxxopts::ParseResult& result, auto& options) -> int
{
    std::cout << "Running memcpy test" << std::endl;
    
    auto gpu_ids = get_gpus(result);
    std::string gpu_ids_str = "";
    for (auto gpu_id : gpu_ids) {
        gpu_ids_str += std::to_string(gpu_id) + ",";
    }
    std::cout << "  GPUs: " << gpu_ids_str << std::endl;
    std::cout << "  Size: " << result["size"].as<std::string>() << std::endl;
    std::cout << "  Duration: " << result["duration"].as<std::string>() << std::endl;
    std::cout << "  Threads per GPU: " << result["threads"].as<int>() << std::endl;
    //std::cout << "  Pattern: " << vector_to_string(result["pattern"].as<std::vector<std::string>>()) << std::endl;
    std::cout << "  Repeat: " << result["repeat"].as<int>() << std::endl;
    std::cout << "  Block Size: " << result["block-size"].as<std::string>() << std::endl;
    std::cout << "  Chunk Size: " << result["chunk-size"].as<std::string>() << std::endl;
    std::cout << "  Non Blocking: " << result["non-blocking"].as<bool>() << std::endl;
    std::cout << "  Zero Buffer: " << result["zero-buffer"].as<bool>() << std::endl;
    std::cout << "  Use Pinned: " << result["use-pinned"].as<bool>() << std::endl;
    std::cout << "  Use Unified: " << result["use-unified"].as<bool>() << std::endl;
    
    std::vector<std::thread> threads;

    bool zero_buffer = result["zero-buffer"].as<bool>();
    bool non_blocking = result["non-blocking"].as<bool>();
    bool pinned = result["use-pinned"].as<bool>();
    bool unified = result["use-unified"].as<bool>();
    unsigned long long duration = get_duration(result);
    int repeat = result["repeat"].as<int>();
    std::vector<std::string> patterns = result["pattern"].as<std::vector<std::string>>();
    unsigned long long size = get_size(result);


    for (auto gpu_id : gpu_ids)
    {
        for (int i = 0; i < result["threads"].as<int>(); i++)
        {
            threads.push_back(std::thread(memcpy_thread_fxn, gpu_id, pinned, unified, zero_buffer, non_blocking, size, repeat, duration, patterns));
        }
    }
    for (auto& thread : threads)
    {
        thread.join();
    }

    return 0;
}

auto run(auto& result, auto& options) -> int
{

    if (result.count("help"))
    {
      return print_help(result, options);
    }

    if (result.count("version"))
    {
        return print_version(result, options);
    }

    if (result.count("list-gpus"))
    {
        return list_gpus(result, options);
    }

    return run_memcpy(result, options);

    return 0;
}


auto main(int argc, char** argv) -> int
{
    cxxopts::Options options("sdc-memcpy", "A command tool for the development and detection of silent data corruption on GPUs");

    options.add_options()
        ("h,help", "Print help")
        ("v,version", "Print version")
        ("g,gpus", "List of GPUs to run the test on", cxxopts::value<std::vector<std::string>>()->default_value("all"))
        ("l,list-gpus", "List all available GPUs")
        ("t,threads", "Number of Host threads to use per GPU", cxxopts::value<int>()->default_value("1"))
        ("s,size", "Size of the buffer to copy", cxxopts::value<std::string>()->default_value("1M"))
        ("d,duration", "Duration of the test", cxxopts::value<std::string>()->default_value("100s"))
        ("p,pattern", "Patterns to use for the copy", cxxopts::value<std::vector<std::string>>()->default_value("alternating"))
        ("r,repeat", "Number of times to repeat the test", cxxopts::value<int>()->default_value("1"))
        ("b,block-size", "Block size for the copy", cxxopts::value<std::string>()->default_value("1MB"))
        ("c,chunk-size", "Chunk size for the copy", cxxopts::value<std::string>()->default_value("1MB"))
        ("n,non-blocking", "Use non-blocking copy", cxxopts::value<bool>()->default_value("false"))
        ("z,zero-buffer", "Zero the buffer before use", cxxopts::value<bool>()->default_value("false"))
        ("u,use-pinned", "Use pinned memory", cxxopts::value<bool>()->default_value("false"))
        ("y,use-unified", "Use unified memory", cxxopts::value<bool>()->default_value("false"))
    ;

    auto result = options.parse(argc, argv);

    auto res = parse_args(result);

    if (res != 0)
    {
      std::cout << options.help() << std::endl;
      exit(res);
    }

    return run(result, options);
}