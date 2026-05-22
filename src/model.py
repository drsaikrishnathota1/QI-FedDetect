"""
model.py
LSTM-based anomaly detection model for IoT network traffic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMAnomalyDetector(nn.Module):
    def __init__(self, input_dim=41, hidden_dim=64, num_layers=2, dropout=0.3):
        super(LSTMAnomalyDetector, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs >= 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy().flatten().astype(int))
    return {
        "accuracy": round(accuracy_score(all_labels, all_preds), 4),
        "f1": round(f1_score(all_labels, all_preds, zero_division=0), 4),
        "precision": round(precision_score(all_labels, all_preds, zero_division=0), 4),
        "recall": round(recall_score(all_labels, all_preds, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(all_labels, all_probs), 4),
    }
