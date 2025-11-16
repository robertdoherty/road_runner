"""Dataset utilities for error prediction inputs."""

import json, numpy as np, torch
from torch.utils.data import Dataset, DataLoader


def soft_target(diag2id: dict[str,int], y_diag: list[tuple[str,float]]|None=None, add_other: bool=True) -> torch.Tensor:
    """Convert a list of diagnostic labels and weights to a soft target tensor."""
    y = np.zeros(len(diag2id), np.float32)
    tot = 0.0
    for d,w in y_diag:
        i = diag2id.get(d)
        if i is not None: y[i] += float(w)
        tot += float(w)
    if add_other and "dx.other_or_unclear" in diag2id:
        # Only add residual weight to "other" category if it exists in vocabulary
        y[diag2id["dx.other_or_unclear"]] += max(0.0, 1.0 - tot)
        tot = 1.0
    if tot > 0: y /= y.sum()
    return torch.from_numpy(y)


class HVACDataset(Dataset):
    
    def __init__(self, path: str, v: dict[str, dict[str, int]]):
        self.rows = [json.loads(l) for l in open(path)]
        self.v = v
    
    
    def _multi_hot(self, toks: list[str], map_: dict[str, int]) -> np.ndarray:
        x = np.zeros(len(map_), np.float32)
        for t in toks: 
            i = map_.get(t); 
            if i is not None: x[i] = 1.0
        return x
    
    
    def _one_hot(self, key: str, map_: dict[str, int], unk: str) -> np.ndarray:
        k = key if key in map_ else unk
        x = np.zeros(len(map_), np.float32); x[map_[k]] = 1.0; return x
    
    
    def __getitem__(self, i: int):
        ex = self.rows[i]; eq = ex["equip"]
        x = np.concatenate([
            self._multi_hot(ex["symptoms_canon"], self.v["symptom2id"]),
            self._one_hot(eq.get("system_type","<unk_system_type>"),  self.v["system_type2id"],  "<unk_system_type>"),
            self._one_hot(eq.get("subtype","<unk_subtype>"), self.v["subtype2id"], "<unk_subtype>"),
            self._one_hot(eq.get("brand","<unk_brand>"),     self.v["brand2id"],   "<unk_brand>")
        ], 0)
        y = soft_target(self.v["diag2id"], ex.get("y_diag"), add_other=True)
        return torch.from_numpy(x), y
    
    def __len__(self) -> int: return len(self.rows)

