# Wafer Anomaly Detection

**Author:** Shaan Ahamed M

This project audits and fixes an anomaly detection model for semiconductor manufacturing. 

The goal is to detect defective silicon wafers using 590 sensor readings collected during manufacturing.

---

## What Was Broken?

The original notebook reported a near-perfect F1 score (0.99), but it had three major silent bugs:

1. **Data Leakage & Artificial Test Set:** It applied SMOTE (an oversampling technique) across the entire dataset before splitting. This created a fake 50/50 test set using copied data points. In reality, only about 6.6% of wafers are defective.
2. **PyTorch Tensor Broadcasting Bug:** The input tensor had shape `(batch, 1, features)` while the reconstruction had shape `(batch, features)`. During loss calculation, PyTorch silently expanded this into a 3D matrix `(batch, batch, features)`, comparing every wafer against every other wafer in the batch instead of itself.
3. **Legacy Machine & Invariant Sensors:** 146 wafers were tested on an older machine where `sensor_0` was set to `0.0` as a flag, and `sensor_1` held error codes (`100`, `200`, `300`). Standardizing these numbers like normal measurements caused massive `-30` sigma errors. Also, 146 sensors had zero variance or were mostly empty.

---

## How It Was Fixed

- **Time-Based Split:** Sorted the data by time and split it into Train (70%), Validation (15%), and Test (15%) so future data never leaks into the past.
- **Natural Defect Rate:** Kept the test set at the real factory defect rate (~3.8% in the test window, 6.6% overall) with zero artificial oversampling.
- **Handled Legacy Machine:** Extracted the legacy machine flag and error codes into separate categorical features, and masked them before scaling continuous sensors.
- **Removed Useless Sensors:** Dropped constant and mostly empty sensors using training data only.
- **Fixed Tensor Shapes:** Ensured all tensor shapes match exactly, stopping the broadcasting bug.
- **Tuned Threshold:** Tuned the classification threshold on the validation set to catch as many real defects as possible.

---

## Results

On the real, untouched test set:
- **Recall:** **77.8%** (catches 7 out of 9 defective wafers)
- **ROC-AUC:** **0.623**
- **Test Defect Rate:** **3.81%** (natural factory prevalence)

The original 0.99 score was an illusion from testing on a balanced, leaked test set. The repaired model gives honest metrics under real manufacturing conditions.

---

## How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Open and run the notebook:
```bash
jupyter notebook wafer_guard_fixed.ipynb
```

---

## Files in This Repository

- `wafer_guard_fixed.ipynb`: The complete, working PyTorch notebook with all outputs and graphs.
- `AUDIT.md`: Detailed engineering audit report explaining all findings.
- `secom_telemetry_modified.csv`: The manufacturing telemetry dataset.
- `requirements.txt`: Python dependencies.
