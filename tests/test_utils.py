import json
import random

import numpy as np
import torch

from piiguard.utils import PROJECT_ROOT, get_device, load_config, set_seed, timed, write_run_record


def test_set_seed_is_deterministic():
    set_seed(7)
    a = (random.random(), np.random.rand(), torch.rand(1).item())
    set_seed(7)
    b = (random.random(), np.random.rand(), torch.rand(1).item())
    assert a == b


def test_get_device_valid():
    assert get_device() in {"cuda", "mps", "cpu"}
    assert get_device("cpu") == "cpu"


def test_timer_measures():
    with timed() as t:
        sum(range(1000))
    assert t.elapsed >= 0


def test_configs_load_and_have_same_keys():
    smoke = load_config(PROJECT_ROOT / "configs/smoke.yaml")
    full = load_config(PROJECT_ROOT / "configs/full.yaml")
    assert set(smoke) == set(full) - {"_config_path"} | {"_config_path"}
    assert smoke["data"]["n_train"] < full["data"]["n_train"]
    assert full["data"]["n_train"] == 5000 and full["data"]["n_val"] == 1000
    assert full["gate"]["vqc"]["n_qubits"] == 8 and full["gate"]["vqc"]["n_layers"] == 3
    assert all(p.startswith("/") for p in smoke["paths"].values())


def test_write_run_record(tmp_path):
    p = write_run_record(tmp_path, "unit", config={"a": 1}, seed=3, metrics={"x": np.float64(0.5)})
    rec = json.loads(p.read_text())
    assert rec["seed"] == 3 and rec["metrics"]["x"] == 0.5
    assert {"git_commit", "device", "timings", "config"} <= set(rec)
