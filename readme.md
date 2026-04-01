# 📡 AI-Based Wireless Signal Detection using Spectrograms

## 🚀 Overview

This project implements an **AI-based spectrum sensing system** that detects the presence of different wireless communication signals such as:

* 📶 5G
* 📡 Wi-Fi
* 🔵 Bluetooth
* 📟 ZigBee

The system works by:

1. Simulating wireless signals
2. Converting them into spectrograms (time-frequency representation)
3. Training a Convolutional Neural Network (CNN)
4. Predicting which signals are present

---

## 🧠 Concept

Wireless signals coexist in the same spectrum and are difficult to distinguish in the time domain.

This project uses:

* **Signal Processing (FFT + STFT)** → Converts signals into spectrograms
* **Machine Learning (CNN)** → Learns patterns in spectrograms

---

## ⚙️ Pipeline

```
Signal Generation
        ↓
Mixed RF Signal
        ↓
Spectrogram (STFT)
        ↓
CNN Model
        ↓
Signal Detection
```

---

## 📂 Project Structure

```
Signal-Analysis-ML/
│
├── signal_gen.py              # Simulates RF signals
├── realistic_signal_gen.py    # Visual spectrum simulation
├── dataset_gen.py             # Creates dataset (spectrograms + labels)
├── dataset.py                 # Loads dataset for training
├── model.py                   # CNN model definition
├── train.py                   # Trains the model
├── eval.py                    # Evaluates performance
├── predict.py                 # Runs signal detection
├── rf_generator.py            # RF signal utilities
│
├── dataset.npz                # Generated dataset (not pushed to Git)
├── rf_model.pth              # Trained model (not pushed to Git)
│
└── README.md
```

---

## 🛠️ Requirements

Install dependencies:

```bash
pip install numpy scipy matplotlib torch scikit-learn
```

---

## ▶️ How to Run the Project

### Step 1 — Generate Signals

```bash
python signal_gen.py
```

This will display:

* Time-domain waveform
* Power Spectral Density (PSD)
* Spectrogram

---

### Step 2 — Generate Dataset

```bash
python dataset_gen.py
```

Creates:

```
dataset.npz
```

Contains:

* `X` → spectrograms
* `Y` → labels

---

### Step 3 — Train the Model

```bash
python train.py
```

Creates:

```
rf_model.pth
```

---

### Step 4 — Evaluate Model

```bash
python eval.py
```

Outputs:

* Precision
* Recall
* F1 Score
* Example prediction

---

### Step 5 — Run Prediction

```bash
python predict.py
```

Example output:

```
========== SIGNAL DETECTION RESULT ==========

Ground Truth:
5G: Present
WiFi: Present
Bluetooth: Absent
ZigBee: Present

Predicted Probabilities:
5G: 0.94
WiFi: 0.89
Bluetooth: 0.21
ZigBee: 0.82

Predicted Signals:
5G: Detected
WiFi: Detected
Bluetooth: Not Detected
ZigBee: Detected
```

---

## 📊 Evaluation Metrics

* **Precision** → How accurate detections are
* **Recall** → How many real signals were detected
* **F1 Score** → Balance between precision & recall

---

## 💡 Key Features

* Simulates realistic wireless signals
* Uses STFT for time-frequency analysis
* Multi-label classification (multiple signals at once)
* End-to-end ML pipeline
* Modular and scalable

---

## ⚠️ Notes

* `dataset.npz` and `rf_model.pth` are large files → excluded from Git
* Signals are **synthetic**, not real RF (can be extended using SDR)

---

## 🔮 Future Work

* Integrate **RTL-SDR / PlutoSDR** for real signal capture
* Improve model accuracy
* Deploy on embedded systems (Jetson / FPGA)
* Real-time spectrum monitoring

---

## 🎓 Applications

* Cognitive Radio
* Spectrum Monitoring
* Interference Detection
* Wireless Network Analysis

---

## 🧾 License

This project is for academic and educational purposes.

---

## 👨‍💻 Author

Your Name
(You can add your GitHub profile here)

---
