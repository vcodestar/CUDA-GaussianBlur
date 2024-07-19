// Chasanis Evangelos cs05058
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cuda_runtime_api.h" 


int getSPcores(cudaDeviceProp devProp) {
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major){
        case 2: // Fermi
            if (devProp.minor == 1) cores = mp * 48;
            else cores = mp * 32;
            break;
        case 3: // Kepler
            cores = mp * 192;
            break;
        case 5: // Maxwell
            cores = mp * 128;
            break;
        case 6: // Pascal
            if ((devProp.minor == 1) || (devProp.minor == 2)) cores = mp * 128;
            else if (devProp.minor == 0) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        case 7: // Volta and Turing
            if ((devProp.minor == 0) || (devProp.minor == 5)) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        case 8: // Ampere
            if (devProp.minor == 0) cores = mp * 64;
            else if (devProp.minor == 6) cores = mp * 128;
            else if (devProp.minor == 9) cores = mp * 128; // ada lovelace
            else printf("Unknown device type\n");
            break;
        case 9: // Hopper
            if (devProp.minor == 0) cores = mp * 128;
            else printf("Unknown device type\n");
            break;
        default:
            printf("Unknown device type\n"); 
            break;
    }
    return cores;
}

/* 
 * Retrieves and prints information for every installed NVIDIA
 * GPU device
 */
void cuinfo_print_devinfo() {
    int num_devs, i;
    cudaDeviceProp dev_prop;

    cudaGetDeviceCount(&num_devs);
    if (num_devs == 0) {
        printf("No CUDA devices found.\n");
        return;
    }

    printf("Found %d CUDA device(s):\n", num_devs);
    for (i = 0; i < num_devs; i++) {
        cudaGetDeviceProperties(&dev_prop, i);

        printf("Device Name: %s\n", dev_prop.name);
        printf("CUDA Compute Capability: %d.%d\n", dev_prop.major, dev_prop.minor);
        printf("CUDA Driver/Runtime Version: %d.%d\n", CUDART_VERSION / 1000, (CUDART_VERSION % 100) / 10);
        printf("Number of SMs: %d\n", dev_prop.multiProcessorCount);
        printf("Total Global Memory: %lu bytes\n", (unsigned long)dev_prop.totalGlobalMem);
        printf("Total Constant Memory: %lu bytes\n", (unsigned long)dev_prop.totalConstMem);
        printf("Shared Memory Per Block: %lu bytes\n", (unsigned long)dev_prop.sharedMemPerBlock);

        printf("Total CUDA Cores: %d\n", getSPcores(dev_prop));
    }
}

int main() {
    cuinfo_print_devinfo();
    return 0;
}
