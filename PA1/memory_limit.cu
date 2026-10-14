#include <iostream>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "GPU Model: " << prop.name << "\n";
    std::cout << "Shared memory per block (Default): " 
              << prop.sharedMemPerBlock / 1024.0 << " KB (" 
              << prop.sharedMemPerBlock << " bytes)\n";
    
    std::cout << "Shared memory per block (Max Opt-in): " 
              << prop.sharedMemPerBlockOptin / 1024.0 << " KB (" 
              << prop.sharedMemPerBlockOptin << " bytes)\n";

    std::cout << "Shared memory per Multiprocessor (SM): " 
              << prop.sharedMemPerMultiprocessor / 1024.0 << " KB (" 
              << prop.sharedMemPerMultiprocessor << " bytes)\n";

    return 0;
}