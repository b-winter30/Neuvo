from dataclasses import dataclass
from typing import Optional

@dataclass
class EvolutionConfig:
    """Configuration for the evolutionary process"""
    mutation_rate: float
    population_size: int
    cloning_rate: float
    max_generations: int
    model_name: str
    fitness_function: str
    evolution_type: str
    gene_value: int = 40
    genotype_length: int = 32
    grammar_file: Optional[str] = None
    eco_mode: bool = False
    tournament_size: int = 5
    max_seq_length: int = 128

    def __post_init__(self):
        self.validate()

    def validate(self):
        """Validate configuration parameters"""
        if not 0 <= self.mutation_rate <= 1:
            raise ValueError("Mutation rate must be between 0 and 1")
        if self.population_size < 2:
            raise ValueError("Population size must be at least 2")
        if not 0 <= self.cloning_rate <= 1:
            raise ValueError("Cloning rate must be between 0 and 1")
        if self.max_generations < 1:
            raise ValueError("Max generations must be at least 1")
        if self.evolution_type.lower() not in ['ga', 'ge']:
            raise ValueError("Evolution type must be either 'ga' or 'ge'")

@dataclass
class TrainingConfig:
    """Configuration for model training"""
    learning_rate: float = 5e-5
    batch_size: int = 16
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    output_dir: str = './results'
    logging_dir: str = './logs'
    per_device_train_batch_size=32,
    gradient_accumulation_steps=2,
    fp16=True  # Enable mixed precision training