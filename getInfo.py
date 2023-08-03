import pycuda.autoinit
import pycuda.driver as drv

def get_gpu_info():
    device = drv.Device(0)  # 0 represents the first GPU, change it to the appropriate GPU index if needed

    # Get device properties
    props = device.get_attributes()

    # Get the number of SMs (CUDA cores)
    num_sm = props[drv.device_attribute.MULTIPROCESSOR_COUNT]

    # Get the size of shared memory per block
    shared_mem_per_block = props[drv.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK]

    # Get the maximum number of threads per block
    max_threads_per_block = props[drv.device_attribute.MAX_THREADS_PER_BLOCK]

    # Get the maximum number of warps per block
    max_warps_per_block = props[drv.device_attribute.MAX_THREADS_PER_MULTIPROCESSOR] // 32

    return num_sm, shared_mem_per_block, max_threads_per_block, max_warps_per_block

if __name__ == "__main__":
    num_sm, shared_mem_per_block, max_threads_per_block, max_warps_per_block = get_gpu_info()
    print(f"Number of SMs on the GPU: {num_sm}")
    print(f"Shared memory per block: {shared_mem_per_block / 1024} KB")
    print(f"Max number of thread per block: {max_threads_per_block}")
    print(f"Max number of warp per block: {max_warps_per_block}")

