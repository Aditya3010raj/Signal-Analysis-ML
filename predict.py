# predict.py
import torch
import numpy as np
from model import SpectrumCNN

SIGNALS = ["5G", "WiFi", "Bluetooth", "ZigBee"]

def predict_from_dataset(dataset_file='dataset.npz', model_file='rf_model.pth', idx=0):
    data = np.load(dataset_file)

    X = data['X']
    Y = data['Y']

    x = X[idx]
    gt = Y[idx]

    # Ensure correct CNN input shape (1,1,H,W)
    if x.ndim == 2:
        x = x[np.newaxis, np.newaxis, :, :]
    elif x.ndim == 3:
        x = x[np.newaxis, :, :, :]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SpectrumCNN(n_classes=4).to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()

    xb = torch.from_numpy(x).to(device).float()

    with torch.no_grad():
        logits = model(xb).cpu().numpy()[0]
        probs = 1 / (1 + np.exp(-logits))

    print("\n========== SIGNAL DETECTION RESULT ==========\n")

    print("Ground Truth:")
    for i, s in enumerate(SIGNALS):
        print(f"{s}: {'Present' if gt[i]==1 else 'Absent'}")

    print("\nPredicted Probabilities:")
    for i, s in enumerate(SIGNALS):
        print(f"{s}: {probs[i]:.3f}")

    print("\nPredicted Signals:")
    for i, s in enumerate(SIGNALS):
        if probs[i] >= 0.5:
            print(f"{s}: Detected")
        else:
            print(f"{s}: Not Detected")

    print("\n=============================================\n")


if __name__ == "__main__":
    predict_from_dataset()