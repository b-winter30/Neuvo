import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, load_metric

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

class CustomClassifier(nn.Module):
    def __init__(self, input_size, num_hidden_layers, hidden_layer_size, num_classes, activation_functions, optimizer):
        super(CustomClassifier, self).__init__()
        self.input_size = input_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_size = hidden_layer_size
        self.num_classes = num_classes
        self.activation_functions = activation_functions

        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_layer_size)
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_layer_size, hidden_layer_size))
        self.fc_out = nn.Linear(hidden_layer_size, num_classes)

        # Choose optimizer
        self.optimizer = optimizer
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        for activation, layer in zip(self.activation_functions, self.hidden_layers):
            x = activation(layer(x))
        x = self.fc_out(x)
        return x

def tokenize_function(example):
    
    return tokenizer(example['text'], padding='max_length', truncation=True)

def run():
    # Example usage
    dataset = load_dataset('ag_news', split='train')
    


    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    # Convert to PyTorch tensors
    input_ids = torch.tensor(tokenized_dataset['input_ids'])
    attention_mask = torch.tensor(tokenized_dataset['attention_mask'])
    labels = torch.tensor(tokenized_dataset['label'])

    # Create DataLoader
    train_dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)


    input_size = len(tokenized_dataset['input_ids'][0])
    num_hidden_layers = 2  # Example: 2 hidden layers
    hidden_layer_size = 64  # Example: 64 nodes per hidden layer
    num_classes = 4  # AG News has 4 classes
    activation_functions = [nn.ReLU(), nn.ReLU()]  # Example activation functions
    optimizer = optim.Adam  # Example optimizer

    model = CustomClassifier(input_size, num_hidden_layers, hidden_layer_size, num_classes, activation_functions, optimizer)

    training_args = TrainingArguments(
        per_device_train_batch_size=32,
        num_train_epochs=3,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy='epoch'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    eval_results = trainer.evaluate()

    print(eval_results)