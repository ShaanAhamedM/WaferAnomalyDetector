"""
Automated Pipeline Integrity & Causality Verification Suite.
"""

import os
import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

CSV_PATH = "secom_telemetry_modified.csv"


def extract_ps_l_features(df, sensor_cols):
    is_ps_l = (df["sensor_0"] == 0.0).astype(float)
    fw_100 = ((is_ps_l == 1.0) & (df["sensor_1"] == 100.0)).astype(float)
    fw_200 = ((is_ps_l == 1.0) & (df["sensor_1"] == 200.0)).astype(float)
    fw_300 = ((is_ps_l == 1.0) & (df["sensor_1"] == 300.0)).astype(float)
    
    ps_l_df = pd.DataFrame({
        "is_legacy_ps_l": is_ps_l.values,
        "fw_err_100": fw_100.values,
        "fw_err_200": fw_200.values,
        "fw_err_300": fw_300.values
    }, index=df.index)
    
    cont_df = df[sensor_cols].copy()
    cont_df.loc[is_ps_l == 1.0, "sensor_0"] = np.nan
    cont_df.loc[is_ps_l == 1.0, "sensor_1"] = np.nan
    return ps_l_df, cont_df


def test_chronological_ordering_and_causality():
    """
    Test 1: Verify strict chronological splitting (t_train_max < t_val_min < t_test_min)
    and confirm zero temporal overlap.
    """
    df = pd.read_csv(CSV_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    n = len(df)
    n_train = int(0.65 * n)
    n_val = int(0.15 * n)
    
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]
    
    t_train_max = train_df["timestamp"].max()
    t_val_min = val_df["timestamp"].min()
    t_val_max = val_df["timestamp"].max()
    t_test_min = test_df["timestamp"].min()
    
    assert t_train_max < t_val_min, f"Temporal leak: Train max {t_train_max} >= Val min {t_val_min}"
    assert t_val_max < t_test_min, f"Temporal leak: Val max {t_val_max} >= Test min {t_test_min}"
    assert len(train_df) + len(val_df) + len(test_df) == 1567


def test_ps_l_sentinel_and_firmware_code_isolation():
    """
    Test 2: Verify all 146 PS-L wafers have sensor_0 == 0.0, firmware codes in {100, 200, 300},
    0.00% defect prevalence, and are masked to NaN before continuous scaling.
    """
    df = pd.read_csv(CSV_PATH)
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    y = (df["label"] == 1).astype(int)
    
    ps_l_df, cont_df = extract_ps_l_features(df, sensor_cols)
    is_ps_l = ps_l_df["is_legacy_ps_l"] == 1.0
    
    assert is_ps_l.sum() == 146, f"Expected 146 PS-L wafers, found {is_ps_l.sum()}"
    assert y[is_ps_l].sum() == 0, f"Expected 0 defects on PS-L wafers, found {y[is_ps_l].sum()}"
    
    # Confirm codes are strictly 100, 200, 300
    ps_l_codes = set(df.loc[is_ps_l, "sensor_1"].unique())
    assert ps_l_codes == {100.0, 200.0, 300.0}
    
    # Verify continuous masking
    assert cont_df.loc[is_ps_l, "sensor_0"].isna().all(), "sensor_0 sentinel not masked to NaN"
    assert cont_df.loc[is_ps_l, "sensor_1"].isna().all(), "sensor_1 error codes not masked to NaN"


def test_delayed_activation_and_zero_nans():
    """
    Test 3: Verify that the 6 delayed-activation sensors are identified as invariant
    in the training set, dropped strictly on training statistics, and produce zero NaNs.
    """
    df = pd.read_csv(CSV_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    
    n_train = int(0.65 * len(df))
    n_val = int(0.15 * len(df))
    
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]
    
    _, tr_cont = extract_ps_l_features(train_df, sensor_cols)
    _, val_cont = extract_ps_l_features(val_df, sensor_cols)
    _, te_cont = extract_ps_l_features(test_df, sensor_cols)
    
    # Identify training invariant sensors
    tr_stds = tr_cont.std()
    zero_std_tr = tr_stds[tr_stds == 0.0].index.tolist()
    
    delayed_sensors = ['sensor_74', 'sensor_206', 'sensor_209', 'sensor_342', 'sensor_347', 'sensor_478']
    for s in delayed_sensors:
        assert s in zero_std_tr, f"Expected {s} to be invariant in training data"
        
    missing_rates = tr_cont.isnull().mean()
    high_missing = missing_rates[missing_rates > 0.50].index.tolist()
    all_nan = tr_stds[tr_stds.isna()].index.tolist()
    drop_cols = sorted(list(set(high_missing + zero_std_tr + all_nan)))
    clean_cols = [c for c in sensor_cols if c not in drop_cols]
    
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    
    X_tr = scaler.fit_transform(imputer.fit_transform(tr_cont[clean_cols]))
    X_val = scaler.transform(imputer.transform(val_cont[clean_cols]))
    X_te = scaler.transform(imputer.transform(te_cont[clean_cols]))
    
    assert not np.isnan(X_tr).any()
    assert not np.isnan(X_val).any()
    assert not np.isnan(X_te).any()
    assert not np.isinf(X_tr).any()
    assert not np.isinf(X_val).any()
    assert not np.isinf(X_te).any()


def test_strict_tensor_shape_alignment_and_negative_assertion():
    """
    Test 4: Verify tensor shape contracts and prove that dimension mismatches
    (e.g., (B, 1, D) broadcasting or (B,) target squeeze) raise an AssertionError.
    """
    B, D_cont, D_psl = 16, 444, 4
    
    # Test shape assertion function
    def assert_shapes(recon, target_x, logits, target_y):
        assert recon.shape == target_x.shape, f"Broadcasting bug! Recon {recon.shape} != target_x {target_x.shape}"
        assert logits.shape == target_y.shape, f"Broadcasting bug! Logits {logits.shape} != target_y {target_y.shape}"
        
    # Valid shapes pass
    valid_recon = torch.randn(B, D_cont)
    valid_target_x = torch.randn(B, D_cont)
    valid_logits = torch.randn(B, 1)
    valid_target_y = torch.randn(B, 1)
    assert_shapes(valid_recon, valid_target_x, valid_logits, valid_target_y)
    
    # Negative test 1: Target_x has extra channel dim (B, 1, D) -> Raises AssertionError
    bad_target_x = valid_target_x.unsqueeze(1) # shape (B, 1, D)
    with pytest.raises(AssertionError):
        assert_shapes(valid_recon, bad_target_x, valid_logits, valid_target_y)
        
    # Negative test 2: Target_y has 1D shape (B,) -> Raises AssertionError
    bad_target_y = valid_target_y.squeeze(-1) # shape (B,)
    with pytest.raises(AssertionError):
        assert_shapes(valid_recon, valid_target_x, valid_logits, bad_target_y)
