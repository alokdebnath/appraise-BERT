import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import pandas as pd
import math
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import os
import itertools

# Custom dataset class (unchanged)
class TextValueDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        text = row['generated_text']
        label = row['suddenness'] - 1  # Convert value to 0-4 classes
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Load dataset (unchanged)
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df

# Create DataLoader (unchanged)
def create_dataloader(df, tokenizer, max_length, batch_size):
    dataset = TextValueDataset(df, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model setup (unchanged)
def create_model(num_labels):
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=num_labels)
    return model

# Evaluation function (unchanged)
def evaluate(model, dataloader, device):
    model.eval()
    predictions, true_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
    
    return accuracy, precision, recall, f1

# Training function (adjusted)
def train(model, train_loader, val_loader, epochs, device, learning_rate):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    model = model.to(device)

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        
        model.train()
        train_loss = 0
        train_loop = tqdm(train_loader, desc="Training")
        
        for batch in train_loop:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        print(f'Average Training Loss: {avg_train_loss:.4f}')
        
        # Evaluate on validation set
        val_accuracy, val_precision, val_recall, val_f1 = evaluate(model, val_loader, device)
        print(f'Validation Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}')
    
    return val_accuracy  # Return the validation accuracy to determine best hyperparameters

# Main function with grid search
def grid_search():
    # Configurations
    num_labels = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load datasets
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    train_df = load_dataset('data/train.csv')
    val_df = load_dataset('data/val.csv')

    # Hyperparameters to test
    learning_rates = [1e-6, 3e-6, 7e-6, 1e-5, 2e-5]
    batch_sizes = [8, 16, 32]
    epoch_choices = [10, 25, 50]
    max_lengths = [64, 128, 256]

    # List to keep track of the best hyperparameters
    best_accuracy = 0
    best_hyperparams = {}

    # Iterate over all combinations of hyperparameters
    for lr, batch_size, epochs, max_length in itertools.product(learning_rates, batch_sizes, epoch_choices, max_lengths):
        print(f"Testing hyperparameters: LR={lr}, Batch Size={batch_size}, Epochs={epochs}, Max Length={max_length}")
        
        # Create dataloaders with current hyperparameters
        train_loader = create_dataloader(train_df, tokenizer, max_length, batch_size)
        val_loader = create_dataloader(val_df, tokenizer, max_length, batch_size)

        # Initialize model
        model = create_model(num_labels)

        # Train the model and get validation accuracy
        val_accuracy = train(model, train_loader, val_loader, epochs, device, lr)

        # Update best hyperparameters if current accuracy is better
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_hyperparams = {
                'learning_rate': lr,
                'batch_size': batch_size,
                'epochs': epochs,
                'max_length': max_length
            }

        print(f"Finished testing LR={lr}, Batch Size={batch_size}, Epochs={epochs}, Max Length={max_length}. Val Accuracy={val_accuracy:.4f}")

    print(f"Best Hyperparameters: {best_hyperparams}, Best Validation Accuracy: {best_accuracy:.4f}")

if __name__ == '__main__':
    grid_search()