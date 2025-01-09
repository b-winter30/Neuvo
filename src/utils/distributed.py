import os
import torch
import torch.distributed as dist
from typing import Optional, Dict, Any
import json
from pathlib import Path
import time
from datetime import datetime

class CheckpointManager:
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def save_checkpoint(self, 
                       state: Dict[str, Any], 
                       generation: int, 
                       metrics: Optional[Dict] = None):
        """Save evolution state and metrics"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = self.checkpoint_dir / f"checkpoint_gen_{generation}_{timestamp}.pt"
        
        # Save evolution state
        checkpoint = {
            'generation': generation,
            'state': state,
            'timestamp': timestamp,
            'metrics': metrics or {}
        }
        
        # Save temporary file first
        temp_path = checkpoint_path.with_suffix('.temp')
        torch.save(checkpoint, temp_path)
        
        # Rename to final filename
        temp_path.rename(checkpoint_path)
        
        # Save readable metrics separately
        if metrics:
            metrics_path = checkpoint_path.with_suffix('.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
                
        return checkpoint_path

    def load_latest_checkpoint(self) -> Optional[Dict]:
        """Load the most recent checkpoint"""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_gen_*.pt"))
        if not checkpoints:
            return None
            
        latest_checkpoint = checkpoints[-1]
        try:
            checkpoint = torch.load(latest_checkpoint)
            return checkpoint
        except Exception as e:
            print(f"Error loading checkpoint {latest_checkpoint}: {e}")
            # Try loading the second latest checkpoint if available
            if len(checkpoints) > 1:
                try:
                    checkpoint = torch.load(checkpoints[-2])
                    return checkpoint
                except Exception as e:
                    print(f"Error loading backup checkpoint: {e}")
            return None

    def cleanup_old_checkpoints(self, keep_last_n: int = 5):
        """Remove old checkpoints, keeping only the most recent n"""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_gen_*.pt"))
        if len(checkpoints) > keep_last_n:
            for checkpoint in checkpoints[:-keep_last_n]:
                try:
                    checkpoint.unlink()
                    # Remove corresponding metrics file if it exists
                    metrics_file = checkpoint.with_suffix('.json')
                    if metrics_file.exists():
                        metrics_file.unlink()
                except Exception as e:
                    print(f"Error removing old checkpoint {checkpoint}: {e}")

class DistributedManager:
    def __init__(self, world_size: int = None):
        self.world_size = world_size or torch.cuda.device_count()
        self.initialized = False

    def setup(self, rank: int):
        """Initialize distributed training"""
        if self.initialized:
            return

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=self.world_size,
                rank=rank
            )
        else:
            dist.init_process_group(
                backend='gloo',
                init_method='env://',
                world_size=self.world_size,
                rank=rank
            )
            
        self.initialized = True

    def cleanup(self):
        """Clean up distributed training"""
        if self.initialized:
            dist.destroy_process_group()
            self.initialized = False

    @staticmethod
    def get_rank() -> int:
        """Get the rank of current process"""
        if not dist.is_initialized():
            return 0
        return dist.get_rank()

    @staticmethod
    def synchronize():
        """Synchronize all processes"""
        if not dist.is_initialized():
            return
        dist.barrier()

class ResourceMonitor:
    def __init__(self):
        self.start_time = time.time()
        
    def get_gpu_memory_usage(self) -> Dict[int, float]:
        """Get GPU memory usage for all available GPUs"""
        memory_usage = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_usage[i] = torch.cuda.memory_allocated(i) / 1024**2  # MB
        return memory_usage
        
    def get_runtime_stats(self) -> Dict[str, float]:
        """Get runtime statistics"""
        return {
            'total_runtime_hours': (time.time() - self.start_time) / 3600,
            'gpu_memory_usage': self.get_gpu_memory_usage()
        }