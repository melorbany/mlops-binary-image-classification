"""Training script with MLflow tracking."""
import json
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import mlflow
import mlflow.pytorch
from datetime import datetime

from src.data.dataset import get_dataloaders
from src.models.cnn import get_model

def load_params():
    """Load training parameters."""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(train_loader), 100.0 * correct / total

def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(val_loader), 100.0 * correct / total

def main():
    params = load_params()
    train_params = params["train"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup MLflow
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("cats-dogs-classification")
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(train_params)
        
        # Load data
        train_loader, val_loader, test_loader = get_dataloaders(
            "data/processed",
            batch_size=train_params["batch_size"]
        )
        
        # Initialize model
        model = get_model(train_params["model_name"]).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=train_params["learning_rate"])
        
        # Training loop
        loss_history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_val_acc = 0.0
        
        for epoch in range(train_params["epochs"]):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            
            print(f"Epoch {epoch+1}/{train_params['epochs']}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Log metrics
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            }, step=epoch)
            
            loss_history["train_loss"].append(train_loss)
            loss_history["val_loss"].append(val_loss)
            loss_history["train_acc"].append(train_acc)
            loss_history["val_acc"].append(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), "artifacts/model.pt")
        
        # Test evaluation
        test_loss, test_acc = validate(model, test_loader, criterion, device)
        print(f"\nTest Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        mlflow.log_metrics({
            "test_loss": test_loss,
            "test_accuracy": test_acc
        })
        
        # Save metrics
        metrics = {
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "best_val_accuracy": best_val_acc
        }
        
        with open("artifacts/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        with open("artifacts/loss_curve.json", "w") as f:
            json.dump(loss_history, f, indent=2)
        
        # Log model artifact
        mlflow.pytorch.log_model(model, "model")
        mlflow.log_artifact("artifacts/model.pt")
        mlflow.log_artifact("artifacts/metrics.json")

if __name__ == "__main__":
    main()
