#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tddexp.io import read_jsonl


PASS_FIELDS = ("hidden_passed", "evalplus_passed", "passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train layer-wise linear probes over saved prompt-end hidden states. "
            "Defaults to grouped CV by task_id to avoid leakage across conditions."
        )
    )
    parser.add_argument(
        "--eval-results",
        action="append",
        required=True,
        help=(
            "Evaluation JSONL file or LiveCodeBench eval directory. Can be repeated. "
            "Directories are read from *_eval_all.jsonl files."
        ),
    )
    parser.add_argument(
        "--states-dir",
        action="append",
        default=[],
        help=(
            "Optional directory used to resolve missing state_path files. Can be repeated. "
            "Existing state_path values are used directly first."
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--probe",
        choices=("success", "condition", "rescue"),
        default="success",
        help=(
            "success: predict pass/fail. condition: predict prompt condition. "
            "rescue: among base-condition failures, predict whether target condition passes."
        ),
    )
    parser.add_argument("--condition-filter", nargs="*", default=None)
    parser.add_argument("--condition-labels", nargs="*", default=None)
    parser.add_argument("--base-condition", default="nl_only")
    parser.add_argument("--target-condition", default="nl_tests")
    parser.add_argument("--group-field", default="task_id")
    parser.add_argument("--pass-field", default=None)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--min-samples", type=int, default=8)
    parser.add_argument("--backend", choices=("sklearn", "torch"), default="sklearn")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-epochs", type=int, default=200)
    parser.add_argument("--torch-lr", type=float, default=1e-2)
    parser.add_argument("--torch-weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--torch-batch-size",
        type=int,
        default=0,
        help="Torch mini-batch size. 0 means full-batch training.",
    )
    parser.add_argument(
        "--random-cv",
        action="store_true",
        help="Use StratifiedKFold instead of grouped CV. Mainly for backward-compatible debugging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_eval_rows([Path(path) for path in args.eval_results])
    if not rows:
        raise SystemExit("No evaluation rows found.")
    print(f"Loaded {len(rows)} eval rows.", flush=True)

    state_index = build_state_index([Path(path) for path in args.states_dir])
    examples = build_examples(rows, args, state_index)
    if len(examples) < args.min_samples:
        raise SystemExit(f"Need at least {args.min_samples} examples; got {len(examples)}.")
    print(f"Built {len(examples)} probe examples for probe={args.probe}.", flush=True)

    X, raw_y, groups, meta = load_probe_arrays(examples, args.group_field)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(raw_y)
    classes = [str(item) for item in label_encoder.classes_]
    if len(classes) < 2:
        raise SystemExit(f"Need at least two classes; got {classes}.")

    cv, n_splits, cv_kind = build_cv(
        y=y,
        groups=groups,
        requested_splits=args.cv_splits,
        random_state=args.random_state,
        random_cv=args.random_cv,
    )
    class_counts = {
        classes[class_idx]: int((y == class_idx).sum())
        for class_idx in range(len(classes))
    }
    print(
        "Probe matrix: "
        f"samples={len(y)}, layers={X.shape[1]}, hidden_size={X.shape[2]}, "
        f"classes={class_counts}, cv={cv_kind}/{n_splits}, backend={args.backend}",
        flush=True,
    )
    layer_results = train_layer_probes(
        X=X,
        y=y,
        groups=groups,
        classes=classes,
        cv=cv,
        cv_kind=cv_kind,
        max_iter=args.max_iter,
        backend=args.backend,
        device=args.device,
        random_state=args.random_state,
        torch_epochs=args.torch_epochs,
        torch_lr=args.torch_lr,
        torch_weight_decay=args.torch_weight_decay,
        torch_batch_size=args.torch_batch_size,
    )

    payload = {
        "probe": args.probe,
        "num_samples": int(len(y)),
        "num_layers": int(X.shape[1]),
        "hidden_size": int(X.shape[2]),
        "classes": classes,
        "class_counts": class_counts,
        "cv": {
            "kind": cv_kind,
            "n_splits": int(n_splits),
            "group_field": args.group_field,
            "num_groups": int(len(set(groups))),
        },
        "backend": {
            "name": args.backend,
            "device": args.device if args.backend == "torch" else None,
            "torch_epochs": args.torch_epochs if args.backend == "torch" else None,
            "torch_lr": args.torch_lr if args.backend == "torch" else None,
            "torch_weight_decay": args.torch_weight_decay if args.backend == "torch" else None,
            "torch_batch_size": args.torch_batch_size if args.backend == "torch" else None,
        },
        "filters": {
            "condition_filter": args.condition_filter,
            "condition_labels": args.condition_labels,
            "base_condition": args.base_condition if args.probe == "rescue" else None,
            "target_condition": args.target_condition if args.probe == "rescue" else None,
            "pass_field": args.pass_field or "auto",
        },
        "layers": layer_results,
        "meta": meta,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_layer_csv(output.with_suffix(".csv"), layer_results)
    print(f"Wrote probe results to {output}")
    print(f"Wrote layer metrics to {output.with_suffix('.csv')}")
    print_best_layers(layer_results)


def load_eval_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        if not path.exists():
            raise SystemExit(f"Eval path does not exist: {path}")
        if path.is_file():
            rows.extend(read_jsonl(path))
            continue
        files = sorted(path.glob("*_eval_all.jsonl"))
        if not files:
            raise SystemExit(f"Eval directory has no *_eval_all.jsonl files: {path}")
        for file_path in files:
            rows.extend(read_jsonl(file_path))
    return rows


def build_state_index(states_dirs: list[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for states_dir in states_dirs:
        if not states_dir.exists():
            continue
        if states_dir.is_file():
            index[states_dir.name] = states_dir
            continue
        for path in states_dir.rglob("*.npz"):
            index.setdefault(path.name, path)
    return index


def build_examples(rows: list[dict], args: argparse.Namespace, state_index: dict[str, Path]) -> list[dict]:
    if args.probe == "rescue":
        return build_rescue_examples(rows, args, state_index)

    examples: list[dict] = []
    allowed_conditions = set(args.condition_filter or [])
    allowed_labels = set(args.condition_labels or args.condition_filter or [])
    for row in rows:
        condition = str(row.get("condition", ""))
        if allowed_conditions and condition not in allowed_conditions:
            continue
        state_path = resolve_state_path(row, state_index)
        if state_path is None:
            continue
        label = condition if args.probe == "condition" else str(int(pass_value(row, args.pass_field)))
        if args.probe == "condition" and allowed_labels and label not in allowed_labels:
            continue
        examples.append({"row": row, "state_path": state_path, "label": label})
    return examples


def build_rescue_examples(rows: list[dict], args: argparse.Namespace, state_index: dict[str, Path]) -> list[dict]:
    pass_by_task_condition: dict[tuple[str, str], bool] = {}
    for row in rows:
        task_id = str(row["task_id"])
        condition = str(row["condition"])
        pass_by_task_condition[(task_id, condition)] = pass_value(row, args.pass_field)

    examples: list[dict] = []
    for row in rows:
        task_id = str(row["task_id"])
        condition = str(row["condition"])
        if condition != args.target_condition:
            continue
        base_key = (task_id, args.base_condition)
        if base_key not in pass_by_task_condition:
            continue
        if pass_by_task_condition[base_key]:
            continue
        state_path = resolve_state_path(row, state_index)
        if state_path is None:
            continue
        examples.append(
            {
                "row": row,
                "state_path": state_path,
                "label": str(int(pass_value(row, args.pass_field))),
            }
        )
    return examples


def resolve_state_path(row: dict, state_index: dict[str, Path]) -> Path | None:
    raw = row.get("state_path")
    if not raw:
        return None
    path = Path(str(raw))
    if path.exists():
        return path
    if path.name in state_index:
        return state_index[path.name]
    return None


def pass_value(row: dict, pass_field: str | None = None) -> bool:
    if pass_field:
        if pass_field not in row:
            raise SystemExit(f"Requested pass field is missing: {pass_field}")
        return bool(row[pass_field])
    for field in PASS_FIELDS:
        if field in row:
            return bool(row[field])
    raise SystemExit("Could not infer pass field; expected one of hidden_passed/evalplus_passed/passed.")


def load_probe_arrays(
    examples: list[dict],
    group_field: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    states = []
    labels = []
    groups = []
    meta = []
    missing = 0
    for example in tqdm(examples, desc="loading states", unit="state"):
        row = example["row"]
        try:
            arr = np.load(example["state_path"])
            states.append(arr["prompt_end_hidden"])
        except Exception:
            missing += 1
            continue
        labels.append(example["label"])
        groups.append(str(row.get(group_field, row.get("task_id", ""))))
        meta.append(
            {
                "task_id": row.get("task_id"),
                "condition": row.get("condition"),
                "label": example["label"],
                "state_path": str(example["state_path"]),
            }
        )
    if missing:
        print(f"Skipped {missing} examples with unreadable state files.", file=sys.stderr)
    if not states:
        raise SystemExit("No readable state files found.")
    return (
        np.stack(states, axis=0),
        np.asarray(labels),
        np.asarray(groups),
        meta,
    )


def build_cv(
    y: np.ndarray,
    groups: np.ndarray,
    requested_splits: int,
    random_state: int,
    random_cv: bool,
):
    class_counts = np.bincount(y)
    n_groups = len(set(groups))
    n_splits = min(requested_splits, int(class_counts.min()), n_groups)
    if n_splits < 2:
        raise SystemExit(
            "Need at least two splits after accounting for class counts and groups; "
            f"class_counts={class_counts.tolist()}, num_groups={n_groups}."
        )
    if random_cv:
        return (
            StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state),
            n_splits,
            "StratifiedKFold",
        )
    try:
        return (
            StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state),
            n_splits,
            "StratifiedGroupKFold",
        )
    except Exception:
        return GroupKFold(n_splits=n_splits), n_splits, "GroupKFold"


def train_layer_probes(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    classes: list[str],
    cv,
    cv_kind: str,
    max_iter: int,
    backend: str,
    device: str,
    random_state: int,
    torch_epochs: int,
    torch_lr: float,
    torch_weight_decay: float,
    torch_batch_size: int,
) -> list[dict]:
    splits = make_cv_splits(cv, y, groups, cv_kind)
    if backend == "torch":
        return train_layer_probes_torch(
            X=X,
            y=y,
            classes=classes,
            splits=splits,
            device=device,
            random_state=random_state,
            epochs=torch_epochs,
            lr=torch_lr,
            weight_decay=torch_weight_decay,
            batch_size=torch_batch_size,
        )
    return train_layer_probes_sklearn(
        X=X,
        y=y,
        classes=classes,
        splits=splits,
        max_iter=max_iter,
    )


def make_cv_splits(cv, y: np.ndarray, groups: np.ndarray, cv_kind: str) -> list[tuple[np.ndarray, np.ndarray]]:
    dummy_X = np.zeros((len(y), 1), dtype=np.float32)
    if "Group" in cv_kind:
        return list(cv.split(dummy_X, y, groups=groups))
    return list(cv.split(dummy_X, y))


def train_layer_probes_sklearn(
    X: np.ndarray,
    y: np.ndarray,
    classes: list[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
    max_iter: int,
) -> list[dict]:
    layer_results: list[dict] = []
    for layer_idx in tqdm(range(X.shape[1]), desc="training layer probes", unit="layer"):
        layer_X = X[:, layer_idx, :]
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=max_iter, class_weight="balanced"),
        )
        prob = cross_val_predict(clf, layer_X, y, cv=splits, method="predict_proba")
        pred = prob.argmax(axis=1)
        result = {
            "layer": int(layer_idx),
            "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro")),
        }
        result["auc"] = safe_auc(y, prob, classes)
        layer_results.append(result)
    return layer_results


def train_layer_probes_torch(
    X: np.ndarray,
    y: np.ndarray,
    classes: list[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
    device: str,
    random_state: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
) -> list[dict]:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Torch backend requires PyTorch to be installed.") from exc

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("Torch backend requested CUDA, but torch.cuda.is_available() is False.")

    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)

    torch_device = torch.device(device)
    y_np = y.astype(np.int64)
    num_classes = len(classes)
    layer_results: list[dict] = []
    for layer_idx in tqdm(range(X.shape[1]), desc="training layer probes", unit="layer"):
        layer_X = X[:, layer_idx, :].astype(np.float32, copy=False)
        prob = train_torch_cv_layer(
            layer_X=layer_X,
            y=y_np,
            splits=splits,
            num_classes=num_classes,
            device=torch_device,
            random_state=random_state + layer_idx,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
        )
        pred = prob.argmax(axis=1)
        result = {
            "layer": int(layer_idx),
            "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro")),
        }
        result["auc"] = safe_auc(y, prob, classes)
        layer_results.append(result)
    return layer_results


def train_torch_cv_layer(
    layer_X: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    num_classes: int,
    device,
    random_state: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
) -> np.ndarray:
    import torch

    probs = np.zeros((len(y), num_classes), dtype=np.float32)
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        torch.manual_seed(random_state + fold_idx)
        X_train = torch.from_numpy(layer_X[train_idx]).to(device)
        X_test = torch.from_numpy(layer_X[test_idx]).to(device)
        y_train = torch.from_numpy(y[train_idx]).to(device)

        mean = X_train.mean(dim=0, keepdim=True)
        std = X_train.std(dim=0, keepdim=True).clamp_min(1e-6)
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        model = torch.nn.Linear(X_train.shape[1], num_classes).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        class_counts = torch.bincount(y_train, minlength=num_classes).float().clamp_min(1.0)
        class_weights = (len(y_train) / (num_classes * class_counts)).to(device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

        train_full_batch(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            X_train=X_train,
            y_train=y_train,
            epochs=epochs,
            batch_size=batch_size,
            seed=random_state + fold_idx,
        )
        with torch.no_grad():
            probs[test_idx] = torch.softmax(model(X_test), dim=1).detach().cpu().numpy()
    return probs


def train_full_batch(
    model,
    optimizer,
    loss_fn,
    X_train,
    y_train,
    epochs: int,
    batch_size: int,
    seed: int,
) -> None:
    import torch

    n = X_train.shape[0]
    use_full_batch = batch_size <= 0 or batch_size >= n
    generator = torch.Generator(device=X_train.device)
    generator.manual_seed(seed)
    for _ in range(epochs):
        if use_full_batch:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(X_train), y_train)
            loss.backward()
            optimizer.step()
            continue

        permutation = torch.randperm(n, device=X_train.device, generator=generator)
        for start in range(0, n, batch_size):
            idx = permutation[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(X_train[idx]), y_train[idx])
            loss.backward()
            optimizer.step()


def safe_auc(y: np.ndarray, prob: np.ndarray, classes: list[str]) -> float | None:
    try:
        if len(classes) == 2:
            return float(roc_auc_score(y, prob[:, 1]))
        return float(roc_auc_score(y, prob, multi_class="ovr", average="macro"))
    except ValueError:
        return None


def write_layer_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["layer", "accuracy", "macro_f1", "auc"])
        writer.writeheader()
        writer.writerows(rows)


def print_best_layers(layer_results: list[dict]) -> None:
    auc_rows = [row for row in layer_results if row.get("auc") is not None]
    if auc_rows:
        best_auc = max(auc_rows, key=lambda row: row["auc"])
        print(f"Best AUC layer: {best_auc['layer']} auc={best_auc['auc']:.4f}")
    best_acc = max(layer_results, key=lambda row: row["accuracy"])
    print(f"Best accuracy layer: {best_acc['layer']} accuracy={best_acc['accuracy']:.4f}")


if __name__ == "__main__":
    main()
