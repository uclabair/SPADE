from __future__ import annotations

import os
import re
import argparse
import logging
from glob import glob
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class ProjectionHead(nn.Module):
    """
    Simple projection MLP. Replace with your repo's ProjectionHead if needed.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int = 512, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ExpertHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = ProjectionHead(embedding_dim=in_dim, hidden_dim=max(512, out_dim*2), out_dim=out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, feat_dim]
        return self.proj(x)


# -------------------------
# Data
# -------------------------

class BagDataset(Dataset):
    """
    Loads a single .npy bag of features per case.
    Each item returns a tensor of shape [feat_dim].
    """
    def __init__(self, arr: np.ndarray):
        assert arr.ndim == 2, f"Expected [N, D] array, got shape {arr.shape}"
        self.arr = arr

    def __len__(self) -> int:
        return self.arr.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"feat": torch.from_numpy(self.arr[idx])}


# -------------------------
# Utils
# -------------------------

def find_best_checkpoints(checkpoint_dir: str, filename_regex: str, num_experts: int) -> List[str]:
    """
    From a directory of checkpoints, pick the best (lowest train loss) per expert id.
    filename_regex must contain capture groups for: epoch, trainloss, proto_id (expert id).
      Example: r"ep([0-9]+)_trloss([0-9]+\\.[0-9]+)_valloss[0-9]+\\.[0-9]+_proto([0-9]+)\\.pt"
    """
    best: Dict[int, Tuple[float, str]] = {}  # proto_id -> (trainloss, path)
    pattern = re.compile(filename_regex)

    for fname in os.listdir(checkpoint_dir):
        m = pattern.search(fname)
        if not m:
            continue
        trainloss = float(m.group(2))
        proto = int(m.group(3))
        path = os.path.join(checkpoint_dir, fname)
        if proto not in best or trainloss < best[proto][0]:
            best[proto] = (trainloss, path)

    paths = []
    for p in range(num_experts):
        if p not in best:
            raise FileNotFoundError(f"No checkpoint matched for expert/proto {p} with regex `{filename_regex}`")
        paths.append(best[p][1])

    return paths


def load_experts(ckpt_paths: List[str], feat_dim: int, out_dim: int, device: torch.device, dtype: torch.dtype) -> List[nn.Module]:
    experts: List[nn.Module] = []
    for p in ckpt_paths:
        model = ExpertHead(in_dim=feat_dim, out_dim=out_dim)
        sd = torch.load(p, map_location="cpu")
        # If your checkpoints wrap the state_dict (e.g., {"model": ...}) adjust here:
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        try:
            model.load_state_dict(sd, strict=False)  # strict=False for safety across minor changes
        except Exception as e:
            logging.warning(f"Non-strict load for {p}: {e}")
            model.load_state_dict(sd, strict=False)
        model.to(device=device, dtype=dtype).eval()
        experts.append(model)
    return experts


def compute_expert_weights_from_centroids(
    feats: torch.Tensor,                    # [B, D]
    centroids: torch.Tensor,                # [K, D]
    centroid_to_expert: torch.Tensor,       # [K], values in [0..E-1]
    tau: float = 0.1,
) -> torch.Tensor:
    """
    1) distances: d_{b,k} = ||x_b - c_k||_2
    2) centroid weights: w_k = softmax(-d_k / tau)
    3) expert weights: sum of centroid weights that map to each expert.

    Returns:
        expert_weights: [B, E]
    """
    # [B, K]
    distances = torch.cdist(feats, centroids, p=2)
    centroid_logits = -distances / max(tau, 1e-6)
    centroid_w = F.softmax(centroid_logits, dim=1)  # [B, K]

    K = centroids.size(0)
    E = int(centroid_to_expert.max().item() + 1)
    # Build a [K, E] one-hot mapping
    eye_E = torch.eye(E, device=centroid_to_expert.device, dtype=centroid_w.dtype)  # [E, E]
    map_K_E = eye_E[centroid_to_expert.long()]  # [K, E]
    # Aggregate centroid weights -> expert weights
    expert_w = centroid_w @ map_K_E  # [B, E]
    # Normalize again for numerical safety
    expert_w = expert_w / (expert_w.sum(dim=1, keepdim=True) + 1e-12)
    return expert_w


def aggregate_experts(
    feats: torch.Tensor,            # [B, D]
    experts: List[nn.Module],
    expert_weights: torch.Tensor,   # [B, E]
) -> torch.Tensor:
    """
    Weighted sum of expert outputs with precomputed expert weights.
    """
    B = feats.size(0)
    E = len(experts)
    assert expert_weights.shape == (B, E), f"Expected weights [B, {E}], got {expert_weights.shape}"

    outputs = []
    with torch.inference_mode():
        for e in experts:
            outputs.append(e(feats))  # each: [B, P]
    # Stack: [E, B, P] -> [B, E, P]
    out = torch.stack(outputs, dim=0).permute(1, 0, 2)
    # Apply weights: [B, E, P] * [B, E, 1] -> [B, P]
    out = (out * expert_weights.unsqueeze(-1)).sum(dim=1)
    return out


# -------------------------
# Main pipeline
# -------------------------

def process_case(
    case_path: str,
    experts: List[nn.Module],
    centroids: torch.Tensor,
    centroid_to_expert: torch.Tensor,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    dtype: torch.dtype,
    tau: float,
) -> np.ndarray:
    """
    Loads one case bag and returns aggregated embeddings as numpy array [N, P].
    """
    feats_np = np.load(case_path)  # [N, D]
    dataset = BagDataset(feats_np.astype(np.float32))  # keep loader stable
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    outputs = []
    for batch in loader:
        x = batch["feat"].to(device=device, dtype=dtype, non_blocking=True)  # [B, D]
        # Compute expert weights from centroid distances
        expert_w = compute_expert_weights_from_centroids(x, centroids, centroid_to_expert, tau=tau)  # [B, E]
        # Weighted sum of experts
        y = aggregate_experts(x, experts, expert_w)  # [B, P]
        outputs.append(y.detach().float().cpu().numpy())

    return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, experts[0].proj.net[-1].out_features), dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Extract softmax-distance weighted expert embeddings.")
    # I/O
    parser.add_argument("--input_bags_dir", type=str, required=True,
                        help="Directory with per-case .npy files (each: [num_tiles, feat_dim]).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to write {case_id}.npy embeddings.")
    # Experts & checkpoints
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory containing expert checkpoints.")
    parser.add_argument("--checkpoint_regex", type=str, required=True,
                        help=r"Regex with groups: epoch, trainloss, proto. Ex: ep([0-9]+)_trloss([0-9]+\.[0-9]+)_valloss[0-9]+\.[0-9]+_proto([0-9]+)\.pt")
    parser.add_argument("--num_experts", type=int, required=True, help="Number of experts/protos (E).")
    # Centroids
    parser.add_argument("--centroids_path", type=str, required=True, help="Path to centroids .npy (shape [K, D]).")
    parser.add_argument("--reassigned_clusters_path", type=str, required=True,
                        help="Path to centroid->expert mapping .npy (shape [K], values in 0..E-1).")
    parser.add_argument("--temperature", type=float, default=0.1, help="Softmax temperature for distances (tau).")
    # Model dims
    parser.add_argument("--feat_dim", type=int, required=True, help="Input Virchow feature dimension (D).")
    parser.add_argument("--proj_dim", type=int, default=256, help="Projected embedding dimension (P).")
    # Runtime
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--half", action="store_true", help="Use float16 on GPU for speed/memory.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--pattern", type=str, default="*.npy", help="Glob for input case files.")
    parser.add_argument("--limit", type=int, default=-1, help="Optional limit on #cases (for quick tests).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = torch.float16 if (args.half and device.type == "cuda") else torch.float32

    # Load centroids and mapping
    centroids = torch.from_numpy(np.load(args.centroids_path)).to(device=device, dtype=dtype)           # [K, D]
    centroid_to_expert = torch.from_numpy(np.load(args.reassigned_clusters_path)).to(device=device)     # [K]
    K, D = centroids.shape
    assert D == args.feat_dim, f"Centroid dim {D} != --feat_dim {args.feat_dim}"

    # Find and load best checkpoints per expert
    ckpts = find_best_checkpoints(args.checkpoint_dir, args.checkpoint_regex, args.num_experts)
    experts = load_experts(ckpts, feat_dim=args.feat_dim, out_dim=args.proj_dim, device=device, dtype=dtype)
    logging.info(f"Loaded {len(experts)} experts from {args.checkpoint_dir}")

    # Iterate cases
    case_paths = sorted(glob(os.path.join(args.input_bags_dir, args.pattern)))
    if args.limit > 0:
        case_paths = case_paths[:args.limit]
    logging.info(f"Found {len(case_paths)} case files.")

    for pth in tqdm(case_paths, desc="Cases"):
        case_id = os.path.splitext(os.path.basename(pth))[0]
        out_path = os.path.join(args.output_dir, f"{case_id}.npy")
        if os.path.exists(out_path):
            continue
        try:
            emb = process_case(
                case_path=pth,
                experts=experts,
                centroids=centroids,
                centroid_to_expert=centroid_to_expert,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
                dtype=dtype,
                tau=args.temperature,
            )
            np.save(out_path, emb)
        except Exception as e:
            logging.exception(f"Failed on {case_id}: {e}")

    logging.info("Done.")


if __name__ == "__main__":
    main()
