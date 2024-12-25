#include <iostream>
#include <cuda_runtime.h>
#include "include/cxxopts.hpp"


#define SDC_MEMCPY_VERSION "v0.1.0"

auto parse_args(auto& result) -> int
{
    
    return 0;
}

auto print_help(auto& result, auto& options) -> int
{
    std::cout << options.help() << std::endl;
    return 0;
}

auto print_version(auto& result, auto& options) -> int
{
    std::cout << SDC_MEMCPY_VERSION << std::endl;
    return 0;
}

auto list_gpus(auto& result, auto& options) -> int
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
        ("s,size", "Size of the buffer to copy", cxxopts::value<std::string>()->default_value("1MB"))
        ("d,duration", "Duration of the test", cxxopts::value<std::string>()->default_value("100s"))
        ("p,pattern", "Patterns to use for the copy", cxxopts::value<std::vector<std::string>>()->default_value("alternating"))
        ("r,repeat", "Number of times to repeat the test", cxxopts::value<std::string>()->default_value("1"))
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