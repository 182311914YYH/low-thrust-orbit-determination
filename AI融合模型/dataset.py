"""Real Starlink Data Loader

Loads gold-layer parquet data: informer_samples, gnn_edges, rtn_residuals.
Parses JSON-formatted features, builds graphs, supports train/val/test split.

Batch format:
  history_features:   (B, N, L, F_value)   float
  history_mask:       (B, N, L)            bool (True=valid)
  physics_future:     (B, N, H, 6)         float
  delta_a_rtn_target: (B, N, H, 3)         float
  edge_index:         (2, E)               long
  edge_attr:          (E, F_edge)           float
  quality_weight:     (B, N, H)            float
  reference_future:   (B, N, H, 6)         float
  norad_ids:          (B, N)               long
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# ---- Feature indices in input_features (14 columns) ----
# [x, y, z, vx, vy, vz, sma, ecc, inc, raan, argp, nu, bstar, time_offset]
FEAT_POS = slice(0, 3)
FEAT_VEL = slice(3, 6)
FEAT_OE = slice(6, 13)   # 7 orbital element features
FEAT_TIME = 13            # time offset column

# Feature count for value embedding
N_VALUE_FEATURES = 14


class StarlinkDataset(Dataset):
    """Real Starlink fusion training dataset."""

    def __init__(
        self,
        data_root: str | Path = "数据收集与预处理/12_数据与实验结果/gold",
        split: str = "train",
        num_nodes: int = 60,
        pred_horizon_h: Optional[int] = None,  # None=both, 12 or 24
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.num_nodes = num_nodes

        # Load split info
        import json as _json
        split_file = self.data_root / "dataset_split" / "dataset_split.json"
        with open(split_file, "r") as f:
            si = _json.load(f)
        self.split_sats = set(si.get(f"{split}_satellites", []))

        # Load samples
        samples_file = self.data_root / "informer_samples" / "informer_samples.parquet"
        self.samples_df = pd.read_parquet(samples_file)

        # Filter by split
        self.samples_df = self.samples_df[
            self.samples_df["norad_id"].isin(self.split_sats)
        ].reset_index(drop=True)

        # Filter by horizon
        if pred_horizon_h is not None:
            self.samples_df = self.samples_df[
                self.samples_df["pred_horizon_h"] == pred_horizon_h
            ].reset_index(drop=True)

        # Label computation mode: \"on-the-fly\" from position residuals
        # Uses endpoint formula: Δa = 2*Δr(T)/T² in RTN coordinates
        self._compute_labels = True

        # Load GNN edges
        edges_file = self.data_root / "gnn_samples" / "gnn_edges.parquet"
        self.edges_df = None
        if edges_file.exists():
            self.edges_df = pd.read_parquet(edges_file)
            # Group by graph_time_utc
            self._edge_groups = {}
            if "graph_time_utc" in self.edges_df.columns:
                for gt, grp in self.edges_df.groupby("graph_time_utc"):
                    self._edge_groups[str(gt)] = grp

        self._norad_to_idx = {}
        all_norads = sorted(self.samples_df["norad_id"].unique())
        for i, nid in enumerate(all_norads):
            self._norad_to_idx[int(nid)] = i

        print(f"[Dataset] {split}: {len(self)} samples, "
              f"{len(all_norads)} satellites, horizon={pred_horizon_h}")

    def __len__(self) -> int:
        return len(self.samples_df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.samples_df.iloc[idx]
        norad = int(row["norad_id"])
        start_time = str(row["start_time_utc"])
        horizon = int(row["pred_horizon_h"])

        # Parse features
        input_feats = np.array(json.loads(row["input_features"]), dtype=np.float32)
        label_feats = np.array(json.loads(row["label_features"]), dtype=np.float32)

        # History: (L, 14), Future ref: (H, 14)
        L, H = input_feats.shape[0], label_feats.shape[0]

        # Value features: all 14 columns
        history_features = torch.from_numpy(input_feats)  # (L, 14)

        # Reference future: first 6 columns = ECI state
        ref_future = torch.from_numpy(label_feats[:, :6])  # (H, 6)

        # Physics future: placeholder (should be precomputed or generated on-the-fly)
        # For Stage C (acceleration-only loss), physics_future not needed
        physics_future = torch.zeros(H, 6)

        # History mask: check for zero rows (missing data)
        mask = torch.ones(L)
        zero_rows = (input_feats[:, :6].sum(axis=1) == 0)
        mask[zero_rows] = 0

        # Load label (Δa_RTN)
        delta_a = torch.zeros(H, 3)
        quality = torch.ones(H)
        if self.labels_df is not None:
            label_key = (norad, start_time, horizon)
            if label_key in self._label_index:
                li = self._label_index[label_key]
                lr = self.labels_df.iloc[li]
                delta_a = torch.tensor([
                    lr["delta_a_R"], lr["delta_a_T"], lr["delta_a_N"]
                ], dtype=torch.float32).T  # (3, H) → (H, 3)
                quality = torch.tensor(lr["quality_weight"], dtype=torch.float32)

        return {
            "norad_id": norad,
            "start_time": start_time,
            "pred_horizon_h": horizon,
            "history_features": history_features,
            "history_mask": mask,
            "physics_future": physics_future,
            "delta_a_rtn_target": delta_a,
            "reference_future": ref_future,
            "quality_weight": quality,
        }

    def get_graph(self, graph_time_utc: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Get edge_index and edge_attr for a graph snapshot.

        If graph_time_utc is None, returns the first available graph.
        Maps norad_ids to local node indices.
        """
        if self.edges_df is None:
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 13))

        if graph_time_utc is not None and graph_time_utc in self._edge_groups:
            grp = self._edge_groups[graph_time_utc]
        elif len(self._edge_groups) > 0:
            grp = list(self._edge_groups.values())[0]
        else:
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 13))

        # Map norad_ids to local indices
        src_ids = grp["source_norad_id"].map(self._norad_to_idx)
        tgt_ids = grp["target_norad_id"].map(self._norad_to_idx)
        valid = src_ids.notna() & tgt_ids.notna()
        src_ids = src_ids[valid].astype(int)
        tgt_ids = tgt_ids[valid].astype(int)

        edge_index = torch.tensor(
            [src_ids.values, tgt_ids.values], dtype=torch.long
        )

        # Edge features
        edge_cols = [
            "relative_distance_m", "relative_speed_mps",
            "delta_raan_rad", "delta_mean_anomaly_rad",
        ]
        edge_feats = []
        for col in edge_cols:
            if col in grp.columns:
                edge_feats.append(grp[col].values[valid.values])
        # Pad to 13 features
        n_edges = valid.sum()
        edge_attr = np.zeros((n_edges, 13), dtype=np.float32)
        for j, vals in enumerate(edge_feats):
            edge_attr[:, j] = vals
        edge_attr[:, 10] = grp["same_plane"].values[valid.values].astype(float)
        edge_attr[:, 11] = grp["same_shell"].values[valid.values].astype(float)

        return edge_index, torch.from_numpy(edge_attr)


def collate_batch(batch: list[dict], dataset: StarlinkDataset) -> dict[str, torch.Tensor]:
    """Collate multiple single-satellite samples into a batch with multi-satellite graph.

    Each sample in batch is from one satellite. We stack them along the N dimension,
    producing (B, N, L, F) tensors. Missing satellites are zero-padded.

    For simplicity in Stage C (single-satellite Informer training), N=1.
    For Stage D (multi-satellite GNN), N=num_nodes.
    """
    B = len(batch)

    # Determine shapes
    L = batch[0]["history_features"].shape[0]
    H = batch[0]["delta_a_rtn_target"].shape[0]

    # For single-satellite batches (Stage C): N=1, B=samples
    history = torch.zeros(B, L, N_VALUE_FEATURES)
    mask = torch.zeros(B, L)
    physics = torch.zeros(B, H, 6)
    delta_a = torch.zeros(B, H, 3)
    ref = torch.zeros(B, H, 6)
    quality = torch.zeros(B, H)
    norads = torch.zeros(B, dtype=torch.long)

    for i, s in enumerate(batch):
        history[i] = s["history_features"]
        mask[i] = s["history_mask"]
        # Pad/truncate physics
        h_actual = min(H, s["physics_future"].shape[0])
        physics[i, :h_actual] = s["physics_future"][:h_actual]
        h_actual = min(H, s["delta_a_rtn_target"].shape[0])
        delta_a[i, :h_actual] = s["delta_a_rtn_target"][:h_actual]
        ref[i, :h_actual] = s["reference_future"][:h_actual]
        quality[i, :h_actual] = s["quality_weight"]
        norads[i] = s["norad_id"]

    # Graph: use dataset.get_graph() for a shared graph (simplified for Stage C)
    edge_index, edge_attr = dataset.get_graph()
    # Expand to batch: each sample uses same graph
    # For Stage C (single-satellite, no GNN), edge_index can be empty
    # For Stage D, each batch has its own graph

    return {
        "history_features": history,       # (B, L, F)
        "history_mask": mask,              # (B, L)
        "physics_future": physics,         # (B, H, 6)
        "delta_a_rtn_target": delta_a,     # (B, H, 3)
        "reference_future": ref,            # (B, H, 6)
        "quality_weight": quality,          # (B, H)
        "edge_index": edge_index,           # (2, E)
        "edge_attr": edge_attr,             # (E, F_e)
        "norad_ids": norads,                # (B,)
    }


def create_real_dataloader(
    data_root: str | Path = "数据收集与预处理/12_数据与实验结果/gold",
    split: str = "train",
    batch_size: int = 16,
    pred_horizon_h: Optional[int] = 12,
    num_workers: int = 0,
) -> DataLoader:
    """Create DataLoader with real Starlink data.

    Args:
        split: "train" | "val" | "test"
        batch_size: samples per batch
        pred_horizon_h: 12 or 24 (None = both)
    """
    ds = StarlinkDataset(
        data_root=data_root,
        split=split,
        pred_horizon_h=pred_horizon_h,
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=lambda batch: collate_batch(batch, ds),
        pin_memory=True,
    )


def test_real_dataloader():
    """Smoke test: load a batch of real data."""
    print("=== Real DataLoader Smoke Test ===")
    ds = StarlinkDataset(
        data_root="数据收集与预处理/12_数据与实验结果/gold",
        split="train",
        pred_horizon_h=12,
    )
    print(f"Dataset size: {len(ds)}")

    if len(ds) == 0:
        print("[WARN] No training samples found. Check split config.")
        return False

    # Load one sample
    s = ds[0]
    print(f"Sample 0: norad={s['norad_id']}, horizon={s['pred_horizon_h']}h")
    print(f"  history: {s['history_features'].shape}")
    print(f"  delta_a: {s['delta_a_rtn_target'].shape}, target RMS: "
          f"R={s['delta_a_rtn_target'][:,0].abs().mean():.3e}, "
          f"T={s['delta_a_rtn_target'][:,1].abs().mean():.3e}, "
          f"N={s['delta_a_rtn_target'][:,2].abs().mean():.3e}")
    print(f"  ref_future: {s['reference_future'].shape}")
    print(f"  quality: mean={s['quality_weight'].mean():.2f}")

    # Test batch
    dl = create_real_dataloader(
        data_root="数据收集与预处理/12_数据与实验结果/gold",
        split="train",
        batch_size=4,
        pred_horizon_h=12,
    )
    batch = next(iter(dl))
    print(f"\nBatch shapes:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {tuple(v.shape)}")

    # Test graph
    ei, ea = ds.get_graph()
    print(f"\nGraph: edges={ei.shape[1]}, edge_attr={ea.shape}")

    print("[PASS] Real DataLoader test OK")
    return True


if __name__ == "__main__":
    test_real_dataloader()
