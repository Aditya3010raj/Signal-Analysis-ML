# eval.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset import SpectrumDataset
from model import SpectrumCNN
from sklearn.metrics import precision_recall_fscore_support

def evaluate(dataset_file='dataset.npz', model_file='rf_model.pth', sample_idx=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ds = SpectrumDataset(dataset_file, split='test')
    model = SpectrumCNN(n_classes=4).to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()

    # evaluate all test set
    X = []
    Y = []
    for i in range(len(test_ds)):
        xb, yb = test_ds[i]
        X.append(xb.numpy()); Y.append(yb.numpy())
    X = np.stack(X)  # (N,1,F,T)
    Y = np.stack(Y)  # (N,4)
    with torch.no_grad():
        xb = torch.from_numpy(X).to(device).float()
        logits = model(xb).cpu().numpy()
        probs = 1/(1+np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(Y, preds, average=None, zero_division=0)
    print("Test metrics:")
    for i, name in enumerate(['5G','WiFi','BT','ZB']):
        print(f"  {name}: P={p[i]:.3f} R={r[i]:.3f} F1={f1[i]:.3f}")

    # plot a sample
    sample_x = X[sample_idx][0]  # (F,T)
    sample_y = Y[sample_idx]
    sample_probs = probs[sample_idx]
    plt.figure(figsize=(10,4))
    plt.imshow(sample_x, aspect='auto', origin='lower')
    plt.colorbar(label='norm dB')
    plt.title(f"Sample {sample_idx} spectrogram (GT: {sample_y}, PredProb: {sample_probs.round(2)})")
    plt.xlabel('Time bins'); plt.ylabel('Freq bins')
    plt.show()

if __name__ == "__main__":
    evaluate()
