# Wafer Anomaly Detection

**Author:** Shaan Ahamed M

This project audits and repairs a machine learning pipeline for detecting semiconductor wafer defects using 590 manufacturing sensors (`secom_telemetry_modified.csv`).

---

## What Was Broken?

The original notebook (`bnypi.ipynb`) reported an F1 score of 0.9915, but that score was an illusion caused by three silent bugs:

1. **Global SMOTE and Data Leakage:** SMOTE was applied across the entire dataset before splitting, creating a synthetic 50/50 balanced test set with duplicated training samples.
2. **PyTorch Tensor Broadcasting:** An input shape mismatch `(batch, 1, features)` vs reconstruction `(batch, features)` caused PyTorch to broadcast into a 3D tensor `(batch, batch, features)`, corrupting reconstruction loss gradients.
3. **Mishandling Legacy Probe Station (PS-L) & Invariant Sensors:** 146 wafers from Legacy Station PS-L had `sensor_0 == 0.0` as a sentinel flag and `sensor_1` holding firmware error codes (`100.0`, `200.0`, `300.0`). Standardizing these numbers produced artificial `-30` sigma outliers. Furthermore, 122 sensors had zero variance in training, and 6 delayed-activation sensors only became active during the test window.

---

## Key Forensic Discoveries

1. **PS-L Defect Rate is 0.00%:** Across all 146 PS-L wafers in the dataset, there are zero defects (0/146), compared to 7.32% (104/1421) on modern probe stations.
2. **6 Delayed-Activation Sensors:** Sensors `sensor_74`, `sensor_206`, `sensor_209`, `sensor_342`, `sensor_347`, and `sensor_478` have a standard deviation of `0.0000` in training data and only activate in the test window. Dropping them using training statistics prevents divide-by-zero explosions at test time.
3. **Sensor Temporal Drift:** Critical process sensors undergo significant drift between early training and late testing (e.g., `sensor_28` shifts by `+0.79` sigma, `sensor_542` shifts by `+0.75` sigma).

---

## How It Was Fixed

- **Chronological Split:** Data is sorted by time and split into Train (65%), Validation (15%), and Test (20%, 314 wafers).
- **Natural Prevalence:** The test set remains untouched at its natural defect prevalence (5.41% in the test window, 6.64% across the full dataset) with zero synthetic samples.
- **PS-L Decoupling:** Decoupled `is_legacy_ps_l` into a binary flag and one-hot encoded firmware codes (`fw_err_100`, `fw_err_200`, `fw_err_300`), masking sensors 0 and 1 to `NaN` before fitting the continuous scaler.
- **Decoupled Architecture:** The autoencoder decoder reconstructs only the 444 clean continuous channels. Latent code $z$, reconstruction error, and PS-L flags feed into the classifier head.
- **Best Checkpoint Restoration:** Used dampened class weighting (`sqrt(N_neg / N_pos)`) and restored the best model checkpoint from Epoch 4 to prevent validation loss explosion.
- **Threshold Calibration:** Tuned $\tau^*$ on the validation PR curve.

---

## Measured Performance

| Metric | Original Notebook | Repaired Model (`wafer_guard_fixed.ipynb`) |
| :--- | :---: | :---: |
| **Test Set Size** | 586 wafers (SMOTE balanced) | **314 wafers** (20% Holdout) |
| **Defect Prevalence** | 50.00% (Fake balance) | **5.41% (Natural test rate)** |
| **Decision Threshold** | 0.5000 | **0.0821 (Calibrated on Val)** |
| **Recall (Sensitivity)** | 1.0000 | **0.9412 (94.1%)** |
| **Defects Caught** | 293 / 293 (Synthetic) | **16 / 17** (Real defects, 1 missed) |
| **F1 Score** | 0.9915 | **0.1159** |
| **Precision** | 0.9832 | **0.0618** |
| **Specificity** | 0.9829 | **0.1818** |
| **ROC-AUC** | 1.0000 | **0.5459** |
| **PR-AUC (AP)** | 1.0000 | **0.0662** (vs 0.0541 baseline) |
| **Confusion Matrix** | TN=288, FP=5, FN=0, TP=293 | **TN=54, FP=243, FN=1, TP=16** |

---

## How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run automated verification tests:
```bash
pytest tests/ -v
```

3. Open and run the notebook:
```bash
jupyter notebook wafer_guard_fixed.ipynb
```

---

## Files in This Repository

- `wafer_guard_fixed.ipynb`: The complete, working PyTorch notebook with all outputs and graphs.
- `AUDIT.md`: Detailed engineering audit report explaining all findings and fixes.
- `tests/test_pipeline_integrity.py`: Automated pytest test suite verifying causality, PS-L isolation, delayed-activation handling, and tensor shapes.
- `secom_telemetry_modified.csv`: Telemetry dataset.
- `requirements.txt`: Python package requirements.
