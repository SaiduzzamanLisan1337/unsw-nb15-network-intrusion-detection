"""Fast research-hygiene, thresholding, and federated-learning tests."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from src import config, data_loader, synthetic  # noqa: E402
from src.federated import make_disjoint_noniid_clients, weighted_fedavg  # noqa: E402
from src.preprocessing import CorrelatedFeatureDropper, build_preprocessor, original_feature_name  # noqa: E402
from src.thresholding import select_threshold_for_fpr  # noqa: E402


def _data(tmp_path, n_train=2500, n_test=1200):
    synthetic.generate(tmp_path, n_train, n_test)
    return data_loader.load_split(tmp_path, protocol="benchmark")


def test_scaler_is_fitted_on_train_only(tmp_path):
    train, test = _data(tmp_path)
    Xtr, _ = data_loader.split_xy(train, "binary")
    Xte, _ = data_loader.split_xy(test, "binary")
    prep = build_preprocessor().fit(Xtr)
    scaler = prep.named_steps["columns"].named_transformers_["num"].named_steps["scale"]
    mean_before = scaler.mean_.copy()
    out_tr, out_te = prep.transform(Xtr), prep.transform(Xte)
    assert np.allclose(scaler.mean_, mean_before)
    cols = prep.get_feature_names_out()
    num_idx = [i for i, c in enumerate(cols) if c.startswith(("log__", "num__"))]
    assert np.abs(out_tr[:, num_idx].mean(axis=0)).max() < 1e-6
    assert np.abs(out_te[:, num_idx].mean(axis=0)).max() > 1e-4


def test_unseen_category_and_nan_inf_are_handled(tmp_path):
    train, test = _data(tmp_path)
    Xtr, _ = data_loader.split_xy(train, "binary")
    Xte, _ = data_loader.split_xy(test, "binary")
    prep = build_preprocessor().fit(Xtr)
    Xte = Xte.copy()
    Xte.loc[Xte.index[0], "proto"] = "never-seen-protocol"
    Xte.loc[Xte.index[1], "dur"] = np.nan
    Xte.loc[Xte.index[2], "sbytes"] = np.inf
    out = prep.transform(Xte.replace([np.inf, -np.inf], np.nan))
    assert np.isfinite(out).all()


def test_clean_drops_identifiers_and_labels_and_normalises_raw_schema():
    raw = pd.DataFrame({
        "SrcIP": ["10.0.0.1", "10.0.0.2"], "DstIP": ["10.0.0.9", "10.0.0.9"],
        "Sport": [1, 2], "Dsport": [80, 81], "id": [1, 2], "dur": [1.0, np.inf],
        "proto": [" TCP", "udp "], "service": ["-", "http"], "state": ["FIN", "INT"],
        "sbytes": ["10", "x"], "attack_cat": [" Backdoors", None], "label": [1, 0],
    })
    df = data_loader.clean(raw)
    feats = data_loader.feature_columns(df)
    for banned in ("srcip", "dstip", "sport", "dsport", "id", "label", "attack_cat"):
        assert banned not in feats
    assert df["attack_cat"].tolist() == ["Backdoor", "Normal"]
    assert not np.isinf(df["dur"].to_numpy(dtype=float)).any()
    assert df["proto"].tolist() == ["tcp", "udp"]
    assert np.isnan(df.loc[1, "sbytes"])


def test_correlation_filter_learns_from_train_only():
    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    train = pd.DataFrame({"a": a, "b": a * 2 + 1, "c": rng.normal(size=500)})
    f = CorrelatedFeatureDropper(0.95).fit(train)
    assert f.drop_cols_ == ["b"]
    test = pd.DataFrame({"a": rng.normal(size=50), "b": rng.normal(size=50), "c": rng.normal(size=50)})
    assert list(f.transform(test).columns) == ["a", "c"]


def test_pipeline_is_picklable_and_predicts(tmp_path):
    train, test = _data(tmp_path)
    Xtr, ytr = data_loader.split_xy(train, "binary")
    Xte, _ = data_loader.split_xy(test, "binary")
    pipe = Pipeline([("prep", build_preprocessor()), ("model", LogisticRegression(max_iter=1000))]).fit(Xtr, ytr)
    pipe2 = pickle.loads(pickle.dumps(pipe))
    assert np.array_equal(pipe.predict(Xte), pipe2.predict(Xte))


def test_original_feature_name_aggregation():
    assert original_feature_name("cat__proto_tcp") == "proto"
    assert original_feature_name("cat__service_http") == "service"
    assert original_feature_name("log__dur") == "dur"


def test_swapped_unsw_filenames_are_corrected_by_row_count(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EXPECTED_TRAIN_ROWS", 6)
    monkeypatch.setattr(config, "EXPECTED_TEST_ROWS", 4)
    cols = ["id", "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes", "attack_cat", "label"]
    six = pd.DataFrame({
        "id": range(1, 7), "dur": [1]*6, "proto": ["tcp"]*6, "service": ["-"]*6, "state": ["FIN"]*6,
        "spkts": [1]*6, "dpkts": [1]*6, "sbytes": [10]*6, "dbytes": [10]*6,
        "attack_cat": ["Normal", "DoS", "DoS", "Normal", "DoS", "Normal"], "label": [0, 1, 1, 0, 1, 0],
    }, columns=cols)
    four = six.iloc[:4].copy()
    six.to_csv(tmp_path / f"{config.TEST_FILE_STEM}.csv", index=False)
    four.to_csv(tmp_path / f"{config.TRAIN_FILE_STEM}.csv", index=False)
    tr, te, meta = data_loader.load_split(tmp_path, protocol="benchmark", return_meta=True)
    assert len(tr) == 6 and len(te) == 4
    assert meta["partition"]["filename_roles_swapped"] is True


def test_research_protocol_removes_train_duplicates_and_overlap(tmp_path):
    train = pd.DataFrame({
        "id": [1, 2, 3, 4], "dur": [1.0, 2.0, 3.0, 2.0], "proto": ["tcp"]*4, "service": ["-"]*4,
        "state": ["FIN"]*4, "sbytes": [10, 20, 30, 20], "attack_cat": ["Normal", "DoS", "DoS", "DoS"],
        "label": [0, 1, 1, 1]
    })
    test = train.iloc[[1, 2]].copy()
    train.to_csv(tmp_path / f"{config.TRAIN_FILE_STEM}.csv", index=False)
    test.to_csv(tmp_path / f"{config.TEST_FILE_STEM}.csv", index=False)
    tr, te, meta = data_loader.load_split(tmp_path, protocol="research", return_meta=True)
    assert len(te) == 2
    assert len(tr) == 1
    assert meta["hygiene"]["train_duplicate_rows_removed"] == 1
    assert meta["hygiene"]["train_test_overlap_rows_removed"] == 2
    assert meta["remaining_train_test_shared_distinct_rows"] == 0


def test_threshold_selection_hits_oof_fpr_cap():
    y = np.array([0] * 900 + [1] * 100)
    # Overlap is deliberately non-trivial; the optimizer must move above 0.5.
    p = np.r_[np.linspace(0.05, 0.65, 900), np.linspace(0.35, 0.99, 100)]
    result = select_threshold_for_fpr(y, p, target_fpr=0.05)
    pred = (p >= result["threshold"]).astype(int)
    fp = int(((y == 0) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    assert fp / (fp + tn) <= 0.05 + 1e-12
    assert result["source"] == "out_of_fold_training_predictions"


def test_federated_clients_are_disjoint_and_complete(tmp_path):
    train, _ = _data(tmp_path, 5000, 1000)
    clients = make_disjoint_noniid_clients(train.reset_index(drop=True), n_clients=4, alpha=0.35, seed=42)
    flat = np.concatenate(clients)
    assert len(flat) == len(train)
    assert len(np.unique(flat)) == len(train)


def test_weighted_fedavg_is_weighted():
    s1 = (np.array([1.0, 3.0]), 1.0)
    s2 = (np.array([3.0, 5.0]), 3.0)
    w, b = weighted_fedavg([s1, s2], [1, 3])
    assert np.allclose(w, np.array([2.5, 4.5]))
    assert np.isclose(b, 2.5)
