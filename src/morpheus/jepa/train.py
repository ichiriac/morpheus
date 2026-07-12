"""Entraînement JEPA (torch). Encode les transitions avec un E_state gelé, entraîne P.

Usage :
    morpheus train-jepa --config configs/jepa.yaml
ou :
    python -m morpheus.jepa.train --config configs/jepa.yaml
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .data import Transition, describe_records, load_transitions
from .encoders import build_encoder


@dataclass
class JepaConfig:
    # données
    source: str = "synthetic"          # synthetic | jsonl:<path> | hf:<name>
    hf_split: str = "train"
    limit: int | None = None
    alfworld: bool = False
    steps_key: str | None = None
    n_episodes: int = 200              # pour source=synthetic
    # encodeur (E_state gelé)
    encoder: str = "hashing"           # hashing | sentence_transformer
    encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    enc_dim: int = 256                 # utilisé par hashing ; sinon déduit de l'encodeur
    # modèle
    latent_dim: int = 256
    action_dim: int = 128
    hidden: int = 512
    # optimisation
    epochs: int = 20
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    val_frac: float = 0.1
    seed: int = 0
    device: str = "auto"               # auto | cpu | cuda
    # pertes
    w_pred: float = 1.0
    w_var: float = 1.0
    w_cov: float = 0.04
    # sortie
    out_dir: str = "checkpoints/jepa"

    @classmethod
    def load(cls, path: str) -> "JepaConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(**data)


def _encode_transitions(enc, transitions: list[Transition]) -> dict[str, np.ndarray]:
    obs = enc.encode([t.obs for t in transitions])
    act = enc.encode([t.action for t in transitions])
    nxt = enc.encode([t.next_obs for t in transitions])
    return {"obs": obs, "act": act, "next": nxt}


def train(cfg: JepaConfig) -> dict:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from .losses import jepa_loss
    from .model import JEPA

    torch.manual_seed(cfg.seed)
    device = ("cuda" if (cfg.device == "auto" and torch.cuda.is_available())
              else ("cpu" if cfg.device == "auto" else cfg.device))

    # 1. données → transitions
    trans = load_transitions(
        cfg.source, hf_split=cfg.hf_split, limit=cfg.limit,
        alfworld=cfg.alfworld, steps_key=cfg.steps_key,
        n_episodes=cfg.n_episodes, seed=cfg.seed,
    )
    trans = [t for t in trans if t.is_valid()]
    if not trans:
        raise RuntimeError("aucune transition valide chargée — vérifie la source/normalisation")
    print(describe_records(trans))

    # 2. encodage (E_state gelé)
    enc = build_encoder(cfg.encoder, dim=cfg.enc_dim, model_name=cfg.encoder_model)
    emb = _encode_transitions(enc, trans)
    enc_dim = emb["obs"].shape[1]

    ds = TensorDataset(
        torch.from_numpy(emb["obs"]), torch.from_numpy(emb["act"]),
        torch.from_numpy(emb["next"]),
    )
    n_val = max(1, int(len(ds) * cfg.val_frac))
    n_train = len(ds) - n_val
    g = torch.Generator().manual_seed(cfg.seed)
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val], generator=g)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size)

    # 3. modèle + optim
    model = JEPA(enc_dim, cfg.latent_dim, cfg.action_dim, cfg.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def run_epoch(dl, train_mode: bool):
        model.train(train_mode)
        agg = {"pred": 0.0, "var": 0.0, "cov": 0.0, "total": 0.0}
        nb = 0
        for s, a, sn in dl:
            s, a, sn = s.to(device), a.to(device), sn.to(device)
            with torch.set_grad_enabled(train_mode):
                pred, target, z = model(s, a, sn)
                loss, logs = jepa_loss(pred, target, z, cfg.w_pred, cfg.w_var, cfg.w_cov)
                if train_mode:
                    opt.zero_grad(); loss.backward(); opt.step()
            for k in agg:
                agg[k] += logs[k]
            nb += 1
        return {k: v / max(1, nb) for k, v in agg.items()}

    history = []
    best_val = float("inf")
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for ep in range(1, cfg.epochs + 1):
        tr = run_epoch(train_dl, True)
        va = run_epoch(val_dl, False)
        history.append({"epoch": ep, "train": tr, "val": va})
        print(f"epoch {ep:>3} | train pred={tr['pred']:.4f} total={tr['total']:.4f} "
              f"| val pred={va['pred']:.4f} var={va['var']:.4f} cov={va['cov']:.4f}")
        if va["pred"] < best_val:
            best_val = va["pred"]
            torch.save(
                {"model": model.state_dict(), "config": asdict(cfg), "enc_dim": enc_dim},
                out / "jepa.pt",
            )

    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False),
                                     encoding="utf-8")
    print(f"\n✓ checkpoint : {out / 'jepa.pt'} | meilleure val pred = {best_val:.4f}")
    return {"best_val_pred": best_val, "n_transitions": len(trans), "enc_dim": enc_dim}


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="morpheus train-jepa")
    p.add_argument("--config", required=True)
    args = p.parse_args(argv)
    train(JepaConfig.load(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
