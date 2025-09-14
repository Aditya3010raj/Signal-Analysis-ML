# predict.py
import torch
import numpy as np
from rf_generator import generate_example
from model import SpectrumCNN

def predict_from_dataset(dataset_file='dataset.npz', model_file='rf_model.pth', idx=0):
    import numpy as np
    data = np.load(dataset_file)
    X = data['X']; Y = data['Y']
    x = X[idx]
    gt = Y[idx]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpectrumCNN(n_classes=4).to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()
    xb = torch.from_numpy(x).unsqueeze(0).to(device).float()
    with torch.no_grad():
        logits = model(xb).cpu().numpy()[0]
        probs = 1/(1+np.exp(-logits))
    print("GT:", gt, "Pred probs:", probs.round(3), "Pred bins:", (probs>=0.5).astype(int))

if __name__ == "__main__":
    predict_from_dataset()
