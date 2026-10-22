"""Train the CNN on DS1 patients, early-stop on held-out DS1 patients, evaluate once on DS2.

Runs several seeds and reports mean +- std, so the numbers are not one lucky run.

    python train.py               # 3 seeds
    python train.py --seeds 1 --epochs 3   # quick smoke test
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data import CLASSES, load_splits
from src.metrics import macro_f1, report, save_report
from src.model import BeatNet

RESULTS = Path(__file__).resolve().parent / "results"


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def loader(d, split, mu, sd, bs, shuffle):
    f = (d[f"{split}_f"] - mu) / sd
    ds = TensorDataset(torch.from_numpy(d[f"{split}_x"]), torch.from_numpy(f.astype(np.float32)),
                       torch.from_numpy(d[f"{split}_y"]))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle)


@torch.no_grad()
def predict(model, dl, dev):
    model.eval()
    return torch.cat([model(x.to(dev), f.to(dev)).argmax(1).cpu() for x, f, _ in dl]).numpy()


def augment(x):
    """Amplitude scaling, baseline wander and noise - cheap ways to look like a different patient."""
    b = x.shape[0]
    scale = 1 + 0.1 * torch.randn(b, 1, 1, device=x.device)
    t = torch.linspace(0, 1, x.shape[-1], device=x.device)
    wander = 0.1 * torch.randn(b, 1, 1, device=x.device) * torch.sin(
        2 * torch.pi * (t + torch.rand(b, 1, 1, device=x.device)))
    return x * scale + wander + 0.05 * torch.randn_like(x)


def train_one(d, args, seed, dev, mu, sd):
    torch.manual_seed(seed)
    np.random.seed(seed)
    counts = np.bincount(d["train_y"], minlength=4).astype(np.float32)
    w = 1.0 / np.sqrt(counts)
    w = torch.tensor(w / w.sum() * len(w), device=dev)

    train_dl = loader(d, "train", mu, sd, args.bs, True)
    val_dl = loader(d, "val", mu, sd, 1024, False)

    model = BeatNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    best, best_state, bad, history = -1.0, None, 0, []
    for ep in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, f, y in train_dl:
            x, f, y = x.to(dev), f.to(dev), y.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(augment(x), f), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        sched.step()
        # validation patients have only 16 F beats, so early stopping uses N/S/V macro-F1
        val_f1 = macro_f1(d["val_y"], predict(model, val_dl, dev), classes=["N", "S", "V"])
        history.append({"epoch": ep, "train_loss": total / len(train_dl.dataset), "val_f1_NSV": val_f1})
        print(f"  seed {seed} epoch {ep:2d}  loss {history[-1]['train_loss']:.4f}  val F1(N,S,V) {val_f1:.3f}")
        if val_f1 > best:
            best, bad = val_f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    return model, best, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    dev = device()
    d = load_splits()
    for s in ["train", "val", "test"]:
        print(f"{s:5s}", dict(zip(CLASSES, np.bincount(d[f'{s}_y'], minlength=4).tolist())))
    mu, sd = d["train_f"].mean(0), d["train_f"].std(0) + 1e-6
    test_dl = loader(d, "test", mu, sd, 1024, False)
    RESULTS.mkdir(exist_ok=True)

    t0 = time.time()
    runs = []
    for seed in range(args.seeds):
        model, best_val, history = train_one(d, args, seed, dev, mu, sd)
        rep = report(d["test_y"], predict(model, test_dl, dev))
        rep["seed"], rep["best_val_f1_NSV"], rep["epochs_run"] = seed, best_val, len(history)
        runs.append((rep, model, history))
        print(f"seed {seed}: DS2 acc {rep['accuracy']:.3f}  macro-F1 {rep['macro_f1']:.3f}  " +
              "  ".join(f"{c} {v['sensitivity']:.2f}/{v['ppv']:.2f}" for c, v in rep["per_class"].items()))

    # The model kept for plots/weights is chosen by VALIDATION score, never by test score.
    rep, model, history = max(runs, key=lambda r: r[0]["best_val_f1_NSV"])
    torch.save(model.state_dict(), RESULTS / "beatnet.pt")
    rep["train_seconds_all_seeds"] = round(time.time() - t0, 1)
    rep["device"] = str(dev)
    save_report(rep, RESULTS, "cnn")
    (RESULTS / "cnn_history.json").write_text(json.dumps(history, indent=2))

    summary = {"accuracy": [r[0]["accuracy"] for r in runs], "macro_f1": [r[0]["macro_f1"] for r in runs]}
    for c in CLASSES:
        for m in ["sensitivity", "ppv"]:
            summary[f"{c}_{m}"] = [r[0]["per_class"][c][m] for r in runs]
    summary = {k: {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4), "runs": v}
               for k, v in summary.items()}
    (RESULTS / "cnn_seeds.json").write_text(json.dumps(summary, indent=2))

    fig, ax1 = plt.subplots(figsize=(5, 3))
    ep = [h["epoch"] for h in history]
    ax1.plot(ep, [h["train_loss"] for h in history], label="train loss")
    ax2 = ax1.twinx()
    ax2.plot(ep, [h["val_f1_NSV"] for h in history], color="tab:orange", label="val F1 (N,S,V)")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax2.set_ylabel("F1")
    fig.legend(loc="upper center", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "cnn_training.png", dpi=150)

    print(f"\nDS2 test (unseen patients), {args.seeds} seeds, mean +- std")
    for k, v in summary.items():
        print(f"  {k:16s} {v['mean']:.3f} +- {v['std']:.3f}")


if __name__ == "__main__":
    main()
