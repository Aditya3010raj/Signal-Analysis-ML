# train.py
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import SpectrumDataset
from model import SpectrumCNN
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from dataset_gen import generate_dataset

def train(args):
    if args.generate:
        print("Generating dataset...")
        generate_dataset(out_file=args.dataset, n_examples=args.n_examples, seed=args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = SpectrumDataset(args.dataset, split='train')
    val_ds = SpectrumDataset(args.dataset, split='val')
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    model = SpectrumCNN(n_classes=4).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
        avg_train_loss = running / len(train_loader)
        # validation
        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                probs = torch.sigmoid(logits).cpu().numpy()
                y_pred.append(probs)
                y_true.append(yb.numpy())
        y_true = np.vstack(y_true)
        y_pred = np.vstack(y_pred)
        y_pred_bin = (y_pred >= 0.5).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred_bin, average=None, zero_division=0)
        print(f"Epoch {epoch+1}/{args.epochs} TrainLoss={avg_train_loss:.4f}")
        for i, name in enumerate(['5G','WiFi','BT','ZB']):
            print(f"  {name}: P={p[i]:.3f} R={r[i]:.3f} F1={f1[i]:.3f}")
    # save
    torch.save(model.state_dict(), args.model_out)
    print("Model saved to", args.model_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="Generate dataset first")
    parser.add_argument("--dataset", default="dataset.npz")
    parser.add_argument("--n_examples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model_out", default="rf_model.pth")
    args = parser.parse_args()
    train(args)
