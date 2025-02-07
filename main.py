import argparse
import os
from datasets import load_dataset
from rich.console import Console
from neuvo import NeuvoBuilderTransformer, EvolutionConfig
import torch

def load_dataset_from_hub(dataset_name: str, subset: str = None):
    """Load dataset from Hugging Face hub"""
    try:
        if subset:
            dataset = load_dataset(dataset_name, subset)
        else:
            dataset = load_dataset(dataset_name)
        return dataset
    except Exception as e:
        console = Console()
        console.print(f"[red]Error loading dataset: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Neuroevolution for Transformer Models"
    )
    
    # Dataset arguments
    parser.add_argument(
        "-d", "--dataset",
        default="glue",
        help="Dataset name from HuggingFace hub"
    )
    parser.add_argument(
        "-ds", "--dataset_subset",
        default="mnli",
        help="Dataset subset (if applicable)"
    )
    
    # Evolution arguments
    parser.add_argument(
        "-t", "--type",
        default='ga',
        choices=['ga', 'ge'],
        help="Type of evolutionary algorithm (ga/ge)"
    )
    parser.add_argument(
        "-e", "--eco",
        action='store_true',
        help="Enable ecological mode"
    )
    parser.add_argument(
        "-ne", "--no-eco",
        dest='eco',
        action='store_false'
    )
    parser.add_argument(
        "-gf", "--grammar_file",
        default=None,
        help="Grammar file for GE (if using GE)"
    )
    
    # Model configuration
    parser.add_argument(
        "-m", "--model",
        default="bert-base-uncased",
        help="Model name from HuggingFace hub"
    )
    parser.add_argument(
        "-ps", "--population_size",
        type=int,
        default=3,
        help="Population size for evolution"
    )
    parser.add_argument(
        "-mg", "--max_generations",
        type=int,
        default=2,
        help="Maximum number of generations"
    )
    parser.add_argument(
        "-mr", "--mutation_rate",
        type=float,
        default=0.1,
        help="Mutation rate"
    )
    parser.add_argument(
        "-cr", "--cloning_rate",
        type=float,
        default=0.2,
        help="Cloning rate"
    )
    parser.add_argument(
        "-v", "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbosity level"
    )
    
    parser.set_defaults(eco=False)
    args = parser.parse_args()

    # Set up console for rich output
    console = Console()
    
    # Load dataset
    with console.status("[bold green]Loading dataset..."):
        dataset = load_dataset_from_hub(args.dataset, args.dataset_subset)
        if dataset is None:
            return

    # Create evolution configuration
    config = EvolutionConfig(
        mutation_rate=0.1,
        population_size=10,
        cloning_rate=0.2,
        max_generations=200,
        model_name='distilbert-base-uncased',
        fitness_function='eval_accuracy',
        evolution_type='ga'
    )

    builder = NeuvoBuilderTransformer(
        config=config,
        verbose=1,
        checkpoint_dir="./checkpoints",
        distributed=torch.cuda.device_count() > 1
    )

    builder.load_data(dataset)
    best_model = builder.run(
        plot=True,
        checkpoint_frequency=5  # Save every 5 generations
    )

    # Save results
    console.print("[bold green]Evolution complete! Results saved.")

if __name__ == '__main__':
    # Set HuggingFace cache directory if needed
    if 'HF_HOME' not in os.environ:
        os.environ['HF_HOME'] = './huggingface_cache'
    
    main()