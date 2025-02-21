import torch
import gc

def flush_gpu_memory():
    """
    Completely flush NVIDIA GPU memory by:
    1. Emptying the GPU cache
    2. Running garbage collection
    3. Resetting the PyTorch CUDA memory allocator
    """
    # Clear any existing tensors from GPU memory
    torch.cuda.empty_cache()
    
    # Run Python garbage collection
    gc.collect()
    
    # Reset the PyTorch CUDA memory allocator
    if torch.cuda.is_available():
        torch.cuda.memory.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # On some systems, this additional step helps
        torch.cuda.synchronize()
    
    # Print memory stats to confirm
    if torch.cuda.is_available():
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"GPU Memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

if __name__ == "__main__":
    flush_gpu_memory()