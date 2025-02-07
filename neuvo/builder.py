from typing import Dict, List, Optional, Union
import random
import copy
import gc
from rich.console import Console
import matplotlib.pyplot as plt
from dataclasses import asdict
import torch
from torch.utils.data import DataLoader

from .model import NeuroevolutionTransformer
from .config import EvolutionConfig, TrainingConfig
from transformers import AutoTokenizer, DataCollatorWithPadding
from neuvo.utils.distributed import CheckpointManager, DistributedManager, ResourceMonitor

class NeuvoBuilderTransformer:
    '''[summary]
    '''
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
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # Create appropriate evolution strategy based on config type
        if config.evolution_type.lower() == 'ga':
            from .evolution import GA
            self.evolution_strategy = GA(asdict(config))
        elif config.evolution_type.lower() == 'ge':
            from .evolution import GE
            self.evolution_strategy = GE(asdict(config))
        else:
            raise ValueError(f"Unknown evolution type: {config.evolution_type}")
        
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

    def evolve_population(self):
        """Use evolution strategy to create next generation"""
        new_population = self.evolution_strategy.evolve_population(
            population=self.population,
            fittest=self.fittest,
            tournament_size=self.config.tournament_size
        )
        return new_population

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
                
                # Evolution steps - using strategy pattern
                self.population = self.evolution_strategy.evolve_population(
                    population=self.population,
                    fittest=self.fittest,
                    tournament_size=self.config.tournament_size
                )
                
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
                
                # Save checkpoint after every generation
                if self.generation % checkpoint_frequency == 0:
                    self.console.print("[cyan]Saving checkpoint...")
                    try:
                        self.save_checkpoint()
                        self.console.print("[green]Checkpoint saved successfully")
                    except Exception as e:
                        self.console.print(f"[red]Error saving checkpoint: {e}")
                
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

    def load_data(self, dataset):
        """Load and prepare dataset for training"""
        if not isinstance(dataset, dict):
            # Convert HuggingFace dataset to expected format
            self.data = {
                'train': dataset['train']
            }
            
            # Handle different validation split names
            if 'validation_matched' in dataset:
                self.data['validation'] = dataset['validation_matched']
            elif 'validation_mismatched' in dataset:
                self.data['validation'] = dataset['validation_mismatched']
            elif 'test' in dataset:
                self.data['validation'] = dataset['test']
            else:
                # If no validation set, use a portion of train
                train_len = len(dataset['train'])
                split_idx = int(train_len * 0.9)  # 90-10 split
                self.data['validation'] = dataset['train'].select(range(split_idx, train_len))
                self.data['train'] = dataset['train'].select(range(split_idx))
        else:
            self.data = dataset
        
        if self.verbose > 0:
            self.console.print(f"[green]Loaded dataset:")
            if 'train' in self.data:
                self.console.print(f"Train size: {len(self.data['train'])}")
            if 'validation' in self.data:
                self.console.print(f"Validation size: {len(self.data['validation'])}")
        
        # Automatically tokenize after loading the data
        self.load_and_tokenize_data(self.data)

    def load_and_tokenize_data(self, data):
        """Tokenize the dataset for model input"""
        max_length = 128  # You can adjust this based on your needs
        
        def preprocess_function(examples):
            # For SST-2, we only have single sentences, not premise/hypothesis pairs
            tokenized = self.tokenizer(
                examples['sentence'],      # SST-2 uses 'sentence' as the input column
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors=None
            )
            
            # Add labels
            tokenized['labels'] = examples['label']
            return tokenized

        # Tokenize the train dataset
        self.data['train'] = data['train'].map(
            preprocess_function,
            batched=True,
            remove_columns=['sentence', 'idx']  # SST-2 specific columns to remove
        )
        
        # Tokenize validation dataset (SST-2 has a single validation set)
        self.data['validation'] = data['validation'].map(
            preprocess_function,
            batched=True,
            remove_columns=['sentence', 'idx']
        )
        
        print("Dataset features after tokenization:", self.data['train'].features)
        print("Sample lengths:", [len(x) for x in self.data['train']['input_ids'][:5]])

    def train_population(self):
        """Train all models in the population using tokenized data."""
        training_config = TrainingConfig()
        print(f"Tokenized data in train pop = {self.data}")
        for i, individual in enumerate(self.population):
            if self.verbose > 0:
                self.console.print(f"Training individual {i+1}/{len(self.population)}")
            
            try:
                # Train model using HuggingFace Trainer
                metrics = individual.train(training_config)
                
                # Update individual's fitness based on evaluation metrics
                individual.fitness = metrics.get(
                    self.config.fitness_function,
                    float('-inf')
                )
                
                if self.verbose > 0:
                    self.console.print(f"Training metrics: {metrics}")
                    
            except Exception as e:
                self.console.print(f"[red]Error training individual {i+1}: {str(e)}")
                raise

    def train_population_distributed(self):
        """Train population using distributed processing"""
        batch_size = 4  # Number of models to train simultaneously
        for i in range(0, len(self.population), batch_size):
            batch = self.population[i:i + batch_size]
            training_config = TrainingConfig()
            
            # Train batch in parallel
            for model in batch:
                metrics = model.train(training_config)
                model.fitness = metrics.get(self.config.fitness_function, float('-inf'))
                
            # Clean up GPU memory after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def initialize_population(self):
        """Initialize the first generation of models."""
        self.console.print("[bold blue]Initializing population...")
        
        for _ in range(self.config.population_size):
            model = NeuroevolutionTransformer(
                config=self.config,
                data=self.data
            )
            self.population.append(model)
        
        # Train initial population
        self.train_population()
        self.update_fittest()
        
        self.console.print(f"[green]Population initialized with {len(self.population)} individuals")
        return self

    def update_fittest(self):
        """Update the fittest individual."""
        current_fittest = max(
            self.population,
            key=lambda x: x.fitness
        )
        
        if not self.fittest or current_fittest.fitness > self.fittest.fitness:
            self.fittest = copy.deepcopy(current_fittest)
            if self.verbose > 0:
                self.console.print(
                    f"[green]New best fitness: {self.fittest.fitness:.4f}"
                )

    def save_best_model(self, path: str):
        """Save the best model found during evolution"""
        if self.fittest:
            self.fittest.save(path)
            self.console.print(f"[green]Best model saved to {path}")
        else:
            self.console.print("[red]No model to save!")