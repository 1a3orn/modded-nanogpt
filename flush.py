import torch
import gc
import os
import subprocess
from typing import Tuple

def check_gpu_status() -> Tuple[bool, str]:
    """
    Check GPU status and CUDA availability
    Returns: (is_available, status_message)
    """
    try:
        # Try to get NVIDIA-SMI output
        nvidia_smi = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if nvidia_smi.returncode != 0:
            return False, "nvidia-smi failed - No NVIDIA GPU driver found"
            
        # Check CUDA availability
        if not torch.cuda.is_available():
            return False, "PyTorch CUDA is not available"
            
        return True, "GPU is available and CUDA is initialized"
    except Exception as e:
        return False, f"Error checking GPU status: {str(e)}"

def flush_gpu_memory():
    """
    Completely flush NVIDIA GPU memory with proper error handling
    """
    # First check GPU status
    gpu_available, status_msg = check_gpu_status()
    print(f"GPU Status: {status_msg}")
    
    if not gpu_available:
        print("\nTroubleshooting steps:")
        print("1. Verify NVIDIA drivers are installed: 'nvidia-smi'")
        print("2. Check CUDA installation: 'nvcc --version'")
        print("3. Verify PyTorch CUDA installation:")
        print(f"   - torch.version.cuda: {torch.version.cuda}")
        print(f"   - CUDA_VISIBLE_DEVICES env var: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
        return

    try:
        # Try to initialize CUDA first
        device = torch.device('cuda')
        torch.cuda.init()
        
        # Clear any existing tensors from GPU memory
        torch.cuda.empty_cache()
        
        # Run Python garbage collection
        gc.collect()
        
        # Reset the PyTorch CUDA memory allocator
        torch.cuda.memory.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        
        # Print memory stats to confirm
        print("\nGPU Memory Status:")
        print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Reserved:  {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        
    except Exception as e:
        print(f"\nError during GPU flush: {str(e)}")
        print("\nPossible solutions:")
        print("1. Restart your Python environment")
        print("2. Run 'sudo nvidia-smi -pm 1' to enable persistent mode")
        print("3. Check if CUDA toolkit matches PyTorch version")
        print("4. Consider rebooting the system")

if __name__ == "__main__":
    flush_gpu_memory()