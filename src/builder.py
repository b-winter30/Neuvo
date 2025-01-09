from typing import Dict, List, Optional, Union
import random
import copy
import gc
from rich.console import Console
import matplotlib.pyplot as plt
from dataclasses import asdict
import torch
from torch.utils.data import DataLoader

from .evolution import EvolutionStrategy
from .model import NeuroevolutionTransformer
from .config import EvolutionConfig, TrainingConfig
from .utils.distributed import CheckpointManager, DistributedManager, ResourceMonitor
import torch.distributed as dist
from pathlib import Path

class NeuvoBuilderTransformer:
    def __init__(
        self,
        config: EvolutionConfig,
        verbose: int = 0,
        checkpoint_dir: str = "checkpoints",
        distributed: bool = False
    ):
        self.config = config
        self.verbose = verbose
        self.console = Console()
        
        # Initialize managers
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        self.distributed_manager = DistributedManager() if distributed else None
        self.resource_monitor = ResourceMonitor()
        
        # Core components
        self.population = []
        self.fittest = None
        self.data = None
        self.evolution_strategy = EvolutionStrategy(
            strategy_type=config.evolution_type,
            config=asdict(config)
        )
        
        # Evolution tracking
        self.generation = 0
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'generation': [],
            'runtime_stats': []
        }

    def get_state_dict(self) -> Dict:
        """Get current state for checkpointing"""
        return {
            'generation': self.generation,
            'population': [model.get_state_dict() for model in self.population],
            'fittest': self.fittest.get_state_dict() if self.fittest else None,
            'history': self.history,
            'config': asdict(self.config)
        }

    def load_state_dict(self, state_dict: Dict):
        """Load state from checkpoint"""
        self.generation = state_dict['generation']
        self.history = state_dict['history']
        
        # Restore population
        self.population = []
        for model_state in state_dict['population']:
            model = NeuroevolutionTransformer(self.config, self.data)
            model.load_state_dict(model_state)
            self.population.append(model)
            
        # Restore fittest if exists
        if state_dict['fittest']:
            self.fittest = NeuroevolutionTransformer(self.config, self.data)
            self.fittest.load_state_dict(state_dict['fittest'])

    def save_checkpoint(self):
        """Save current state to checkpoint"""
        state_dict = self.get_state_dict()
        metrics = {
            'current_best_fitness': self.fittest.fitness if self.fittest else None,
            'current_avg_fitness': sum(m.fitness for m in self.population) / len(self.population),
            **self.resource_monitor.get_runtime_stats()
        }
        
        self.checkpoint_manager.save_checkpoint(
            state_dict,
            self.generation,
            metrics
        )
        
        # Cleanup old checkpoints
        self.checkpoint_manager.cleanup_old_checkpoints()

    def resume_from_checkpoint(self) -> bool:
        """Resume evolution from latest checkpoint"""
        checkpoint = self.checkpoint_manager.load_latest_checkpoint()
        if checkpoint:
            self.console.print(f"[green]Resuming from generation {checkpoint['generation']}")
            self.load_state_dict(checkpoint['state'])
            return True
        return False

    def train_population_distributed(self):
        """Train population using distributed processing"""
        if not self.distributed_manager:
            self.train_population()
            return
            
        # Split population among available GPUs
        world_size = self.distributed_manager.world_size
        rank = self.distributed_manager.get_rank()
        
        # Calculate indices for this process
        indices = range(rank, len(self.population), world_size)
        local_population = [self.population[i] for i in indices]
        
        # Train local models
        training_config = TrainingConfig()
        for model in local_population:
            metrics = model.train(training_config)
            model.fitness = metrics.get(self.config.fitness_function, float('-inf'))
            
        # Synchronize results
        self.distributed_manager.synchronize()

    def run(self, plot: bool = True, checkpoint_frequency: int = 5):
        """Run evolution with checkpointing and distribution support"""
        self.console.print("[bold blue]Starting evolution...")
        
        # Try to resume from checkpoint
        if not self.resume_from_checkpoint():
            self.initialize_population()
        
        try:
            while self.generation < self.config.max_generations:
                self.generation += 1
                
                if self.verbose > 0:
                    self.console.print(f"\n[bold]Generation {self.generation}")
                
                # Evolution steps
                parents = self.tournament_selection()
                self.population = self.crossover_and_mutation(parents)
                
                # Distributed training
                if self.distributed_manager:
                    self.train_population_distributed()
                else:
                    self.train_population()
                
                # Update statistics
                self.update_fittest()
                avg_fitness = sum(m.fitness for m in self.population) / len(self.population)
                
                # Update history
                self.history['generation'].append(self.generation)
                self.history['best_fitness'].append(self.fittest.fitness)
                self.history['avg_fitness'].append(avg_fitness)
                self.history['runtime_stats'].append(
                    self.resource_monitor.get_runtime_stats()
                )
                
                # Checkpoint if needed
                if self.generation % checkpoint_frequency == 0:
                    self.save_checkpoint()
                
                # Clean up GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                if self.verbose > 0:
                    self.console.print(
                        f"Best fitness: {self.fittest.fitness:.4f}, "
                        f"Avg fitness: {avg_fitness:.4f}"
                    )
            
            if plot:
                self.plot_evolution()
                
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Evolution interrupted by user")
            self.save_checkpoint()
            if plot:
                self.plot_evolution()
        except Exception as e:
            self.console.print(f"[red]Error during evolution: {e}")
            self.save_checkpoint()
            raise
        finally:
            if self.distributed_manager:
                self.distributed_manager.cleanup()
        
        return self.fittest

    def plot_evolution(self):
        """Plot the evolution progress"""
        plt.figure(figsize=(10, 6))
        plt.plot(
            self.history['generation'],
            self.history['best_fitness'],
            'b-',
            label='Best Fitness'
        )
        plt.plot(
            self.history['generation'],
            self.history['avg_fitness'],
            'r--',
            label='Average Fitness'
        )
        plt.xlabel('Generation')
        plt.ylabel(f'Fitness ({self.config.fitness_function})')
        plt.title('Evolution Progress')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'evolution_progress_{self.config.evolution_type}.png')
        plt.close()

    def save_best_model(self, path: str):
        """Save the best model found during evolution"""
        if self.fittest:
            self.fittest.save(path)
            self.console.print(f"[green]Best model saved to {path}")
        else:
            self.console.print("[red]No model to save!")