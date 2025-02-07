import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM
)

class NeuroevolutionTransformer:
    def __init__(self, config, data, genotype=None, fittest=None):
        self.config = config
        self.data = data
        self.genotype = genotype
        self.fittest = fittest
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_model()

    def setup_model(self):
        """Initialize the model and tokenizer"""
        # Select appropriate model class based on model type
        if "gpt" in self.config.model_name.lower():
            self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
        elif "t5" in self.config.model_name.lower():
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_name)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name, num_labels=3
            )
            
        self.model.to(self.device)

    def train(self, training_config):
        """Train the model using current hyperparameters"""
        training_args = TrainingArguments(
            output_dir=training_config.output_dir,
            num_train_epochs=training_config.num_epochs,
            per_device_train_batch_size=training_config.batch_size,
            per_device_eval_batch_size=training_config.batch_size,
            warmup_steps=training_config.warmup_steps,
            weight_decay=training_config.weight_decay,
            logging_dir=training_config.logging_dir,
            logging_steps=training_config.logging_steps,
            learning_rate=training_config.learning_rate,
            max_grad_norm=training_config.max_grad_norm,
            save_total_limit=1,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        print (f"self.data: {self.data}")
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.data['train'],
            eval_dataset=self.data['validation']
        )

        # Train and evaluate
        trainer.train()
        metrics = trainer.evaluate()
        
        return metrics

    def get_hyperparameters(self):
        """Get current hyperparameters from the model"""
        return {
            'learning_rate': self.model.config.learning_rate,
            'batch_size': self.model.config.batch_size,
            'num_epochs': self.model.config.num_train_epochs,
            # Add other hyperparameters as needed
        }

    def set_hyperparameters(self, hyperparameters):
        """Set new hyperparameters for the model"""
        for name, value in hyperparameters.items():
            setattr(self.model.config, name, value)