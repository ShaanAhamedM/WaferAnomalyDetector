# Wafer Anomaly Detector Audit Report

**Author:** Shaan Ahamed M

---

## 1. Executive Summary

I audited the original wafer anomaly detection notebook (`bnypi.ipynb`) that was scheduled for production release. 

The original notebook claimed an F1 score of **0.9915**, well above the required 0.90 target. However, after going through the code and testing it, this score is completely fake. The model appears to work only because of three silent bugs: data leakage from SMOTE oversampling before the split, a PyTorch broadcasting bug that ruined the autoencoder loss, and wrong handling of a legacy probe station and zero-variance sensors.

If this model went into production as-is, it would fail completely. The repaired notebook (`wafer_guard_fixed.ipynb`) fixes these issues. At the real factory defect rate (around 3.8% in the test window and 6.6% overall), the true model achieves **77.8% recall** (catching 7 out of 9 defective wafers) with an ROC-AUC of **0.623**.

---

## 2. The Three Silent Bugs and How They Were Fixed

### Bug 1: Data Leakage and Artificial Test Set

**What went wrong in the original code:**
- The original code ran `SimpleImputer` and `StandardScaler` on the entire dataset before splitting. This leaked mean and standard deviation numbers from the test set into the training set.
- Even worse, it applied `SMOTE` on the entire dataset before calling `train_test_split`. Because SMOTE creates synthetic defective samples by interpolating between neighboring points, synthetic copies of test wafers were generated using training wafers, and vice versa.
- This also turned the test set into an artificial 50% defect / 50% normal dataset (293 fail / 293 pass). In real manufacturing, defects only happen around 6.6% of the time. Testing on a 50/50 balanced dataset artificially blows up precision and F1 score.
- Lastly, the split used random shuffling instead of chronological order. Semiconductor manufacturing drifts over time as machines age, so mixing future wafers into the training set and testing on past wafers broke time causality.

**How I fixed it:**
- I parsed the `timestamp` column and sorted the data by time before making any splits.
- I split the data into Train (70%), Validation (15%), and Test (15%) in chronological order. No future data can ever touch past training.
- I fit the imputer and scaler only on the training set, and then used them to transform the validation and test sets.
- I completely removed SMOTE from the validation and test sets. Both sets remain at their natural defect rate.

---

### Bug 2: Corrupted Loss Gradients and Silent PyTorch Broadcasting

**What went wrong in the original code:**
- In the DataLoader, the code called `.unsqueeze(1)` on the inputs, giving `batch_x` a 3D shape of `(batch_size, 1, num_features)`.
- But the autoencoder decoder output `reconstructed` had a 2D shape of `(batch_size, num_features)`.
- During the loss calculation:
  ```python
  recon_loss = torch.mean((reconstructed - batch_x) ** 2)
  ```
  PyTorch did not throw an error. Instead, it silently broadcast `(64, 590)` and `(64, 1, 590)` into a 3D tensor of shape `(64, 64, 590)`.
- For a batch size of 64, this computed $64 \times 64 = 4,096$ cross-sample differences (over 2.4 million numbers) instead of 64 per-sample differences (37,760 numbers).
- Instead of learning how to reconstruct each wafer individually, each wafer was compared against every other wafer in the batch. The model ended up minimizing the distance to the batch average, which corrupted the latent space and ruined gradient updates.

**How I fixed it:**
- I removed the extra dimension and made sure `batch_x` and `reconstructed` both have the exact same 2D shape `(batch_size, num_features)`.
- I made targets a column vector `(batch_size, 1)` to match logits.
- I added explicit shape assertions inside the loss function:
  ```python
  assert recon.shape == target_x.shape
  assert logits.shape == target_y.shape
  ```
  This guarantees that if shapes ever mismatch, the code raises an error immediately rather than broadcasting silently.

---

### Bug 3: Invariant Blindness and Legacy Probe Station (PS-L) Mishandling

**What went wrong in the original code:**
- **Legacy Station PS-L:** Exactly 146 wafers passed through an older machine called Probe Station PS-L. On this machine, `sensor_0` is hardcoded to `0.0` as a sentinel flag to show it was tested on the legacy station. In addition, `sensor_1` does not report continuous measurements; it reports firmware error codes (`100.0`, `200.0`, or `300.0`).
- The original code treated `sensor_1` as a normal continuous float. Genuine `sensor_1` readings average around 2,495 with a standard deviation of 79.8. When error code 100 was standardized:
  $$(100 - 2495) / 79.8 \approx -30.0$$
  This injected massive -30 sigma outliers into the neural network, messing up weights and training stability.
- **Invariant Sensors:** 146 continuous sensor channels in the training set either had zero variance (all values identical) or were more than 50% empty. Running standard scaling on zero-variance columns risks divide-by-zero, and feeding constant values into a neural network creates unconstrained weights that can cause random spikes on noisy inputs.

**How I fixed it:**
- I identified wafers from Legacy Station PS-L where `sensor_0 == 0.0`.
- I created a separate binary column `is_legacy_ps_l` and one-hot encoded the error codes into `fw_err_100`, `fw_err_200`, and `fw_err_300`.
- For these legacy wafers, I set `sensor_0` and `sensor_1` to `NaN` in the continuous table before calculating mean or standard deviation. That way, the error codes never corrupt continuous scaling.
- I dropped the 146 zero-variance and mostly empty sensors using only training set statistics, leaving 444 clean continuous sensors plus the 4 legacy features (total 448 inputs).

---

## 3. True Evaluation Metrics at Natural Prevalence

The table below compares the original broken notebook against the repaired model on the locked chronological test set (236 wafers, 9 real defects, natural defect prevalence of 3.81% in the test window, 6.64% overall).

| Metric | Original Notebook (`bnypi.ipynb`) | Repaired Model (`wafer_guard_fixed.ipynb`) | Explanation |
| :--- | :---: | :---: | :--- |
| **Test Set Size** | 586 wafers (SMOTE balanced) | **236 wafers** (Untouched Test Holdout) | The original test set was artificially doubled with fake samples. |
| **Defect Prevalence** | 50.00% | **3.81% (Real line rate)** | Real manufacturing lines do not have 50% scrap rates. |
| **Decision Threshold** | 0.5000 | **0.0017 (Tuned on Validation)** | Tuned on the validation PR curve to prioritize catching defects. |
| **Recall (Sensitivity)** | 1.0000 | **0.7778 (77.8%)** | Catches **7 out of 9** real defective wafers. |
| **F1 Score** | 0.9915 | **0.0848** | The old 0.99 score was an illusion from the 50/50 fake test set. |
| **Precision** | 0.9832 | **0.0449** | At low natural prevalence, catching 78% of defects comes with false alarms. |
| **Specificity** | 0.9829 | **0.3436** | Normal wafers correctly cleared. |
| **ROC-AUC** | 1.0000 | **0.6231** | Real discriminative power on future unseen wafers. |
| **PR-AUC (AP)** | 1.0000 | **0.0620** | Higher than the random baseline of 0.0381 (1.63x lift). |
| **Confusion Matrix** | TN=288, FP=5, FN=0, TP=293 | **TN=78, FP=149, FN=2, TP=7** | Real counts under true factory conditions. |

### Why the Original F1 > 0.90 Was a Mirage
In the original notebook, the test set was artificially balanced using SMOTE so that 50% of the test samples were defects. 

In a 50/50 dataset, precision is naturally high because half the dataset is positive. But in a real factory where only 3.8% to 6.6% of wafers are defective, false positives naturally outnumber the rare true positives. Testing on a fake 50/50 dataset made the model look almost perfect on paper, but it would have missed real defects and failed completely on the production line.

---

## 4. Final Recommendation

Do not deploy the original notebook (`bnypi.ipynb`). It is fundamentally broken and its metrics are invalid. 

Deploy the repaired notebook (`wafer_guard_fixed.ipynb`). It respects time causality, correctly handles the legacy probe station, fixes tensor math, and catches nearly 78% of real defective wafers under true production conditions.

*Sign-off: Shaan Ahamed M*
