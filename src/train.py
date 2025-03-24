from models.fair_hiring_model import FairHiringModel
from data.generate_data import generate_synthetic_data
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, f1_score



input_dim = 20
lambda_adv = 1.0  # Adversarial strength
batch_size = 64
epochs = 50
lr = 0.001


def create_dataloaders(X_train,
                        y_train,
                        s_train,
                        X_test,
                        y_test,
                        s_test,
                        batch_size=64):


    # Convert to PyTorch tensors
    train_data = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train).unsqueeze(1),
        torch.FloatTensor(s_train).unsqueeze(1)
    )
    
    test_data = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test).unsqueeze(1),
        torch.FloatTensor(s_test).unsqueeze(1)
    )

    return (
        DataLoader(train_data, batch_size=batch_size, shuffle=True),
        DataLoader(test_data, batch_size=batch_size)
    )




def train_model(model,
                train_loader):


    # Initialize model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FairHiringModel(input_dim, lambda_adv).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)


    #Training Loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_X, batch_y, batch_s in train_loader:
            batch_X, batch_y, batch_s = (
                batch_X.to(device),
                batch_y.to(device),
                batch_s.to(device)
            )
            optimizer.zero_grad()
            # Forward pass
            primary_pred, sensitive_pred = model(batch_X)

            # Compute loss
            loss_primary = criterion(primary_pred, batch_y)
            loss_sensitive = criterion(sensitive_pred, batch_s)
            loss = loss_primary + loss_sensitive
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
        # Calculate average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
    
 def evaluate_model(model, test_loader):   # --- Evaluation ---
    model.eval()
    primary_preds, sensitive_preds = [], []
    y_true, s_true = [], []
    
    with torch.no_grad():
        for batch_X, batch_y, batch_s in test_loader:
            batch_X = batch_X.to(device)
            p_pred, s_pred = model(batch_X)
            
            primary_preds.extend(p_pred.cpu().numpy())
            sensitive_preds.extend(s_pred.cpu().numpy())
            y_true.extend(batch_y.numpy())
            s_true.extend(batch_s.numpy())
        
        # Calculate metrics
        primary_preds = np.array(primary_preds).flatten()
        sensitive_preds = np.array(sensitive_preds).flatten()
        y_true = np.array(y_true).flatten()
        s_true = np.array(s_true).flatten()
        
        # Binarize predictions
        primary_preds_bin = (primary_preds > 0.5).astype(int)
        sensitive_preds_bin = (sensitive_preds > 0.5).astype(int)
        
        # Calculate accuracy
        primary_acc = accuracy_score(y_true, primary_preds_bin)
        sensitive_acc = accuracy_score(s_true, sensitive_preds_bin)
        
        # Calculate F1 score
        primary_f1 = f1_score(y_true, primary_preds_bin)
        sensitive_f1 = f1_score(s_true, sensitive_preds_bin)
        
        # Print results
        print(f"Primary Accuracy: {primary_acc:.4f}, Sensitive Accuracy: {sensitive_acc:.4f}")
        print(f"Primary F1 Score: {primary_f1:.4f}, Sensitive F1 Score: {sensitive_f1:.4f}")
        
        

def main():
    (X_train, y_train, s_train), (X_test, y_test, s_test) = generate_synthetic_data()
    train_loader, test_loader = create_dataloaders(X_train, y_train, s_train, X_test, y_test, s_test, batch_size)
    model = FairHiringModel(input_dim, lambda_adv).to(device)
    train_model(model, train_loader)
    evaluate_model(model, test_loader)


if __name__ == "__main__":
    main()
