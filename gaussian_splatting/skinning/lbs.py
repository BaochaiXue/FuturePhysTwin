from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from gaussian_splatting.dynamic_utils import (
    calc_weights_vals_from_indices,
    interpolate_motions_speedup,
    knn_weights_sparse,
)


def build_skinning_weights(
    canonical_xyz: torch.Tensor,
    bones: torch.Tensor,
    k: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute sparse skinning weights for canonical Gaussians.

    Args:
        canonical_xyz: (`N`, 3) tensor of Gaussian centres in canonical pose.
        bones: (`B`, 3) tensor of bone anchor positions in canonical pose.
        k: Number of nearest bones to bind each Gaussian to.

    Returns:
        indices: (`N`, `k`) long tensor with the indices of the nearest bones.
        weights: (`N`, `k`) float tensor with normalised skinning weights.
    """

    if canonical_xyz.ndim != 2 or canonical_xyz.shape[1] != 3:
        raise ValueError("canonical_xyz must have shape (N, 3)")
    if bones.ndim != 2 or bones.shape[1] != 3:
        raise ValueError("bones must have shape (B, 3)")
    if k <= 0:
        raise ValueError("k must be a positive integer")

    weights_vals, indices = knn_weights_sparse(bones, canonical_xyz, K=k)
    # knn_weights_sparse returns inverse-distance values; recompute a numerically
    # stable weight normalisation using helper already present in the repo.
    weights = calc_weights_vals_from_indices(bones, canonical_xyz, indices)

    return indices.long(), weights.float()


@dataclass
class LBSDeformer:
    """
    Lightweight linear-blend skinning front-end for Gaussian kernels.

    The class keeps the canonical bone configuration, bone adjacency graph, and
    precomputed skinning indices/weights. At runtime it maps canonical Gaussian
    centres/rotations to a posed configuration using bone displacements.
    """

    bones0: torch.Tensor
    relations: torch.Tensor
    skin_indices: torch.Tensor
    skin_weights: torch.Tensor

    def __post_init__(self) -> None:
        for name in ("bones0", "relations", "skin_indices", "skin_weights"):
            tensor = getattr(self, name)
            if not torch.is_tensor(tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            tensor = tensor.detach().clone()
            if name in ("relations", "skin_indices"):
                tensor = tensor.long()
            else:
                tensor = tensor.float()
            tensor.requires_grad_(False)
            setattr(self, name, tensor)

    @torch.no_grad()
    def deform(
        self,
        xyz0: torch.Tensor,
        quat0: Optional[torch.Tensor],
        motions_t: torch.Tensor,
        *,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply LBS to canonical Gaussian centres/rotations.

        Args:
            xyz0: (`N`, 3) canonical Gaussian centres.
            quat0: (`N`, 4) canonical Gaussian rotations as quaternions. Can be
                ``None`` if rotations are not needed.
            motions_t: (`B`, 3) bone displacements for the target frame.
            device: optional device override. If omitted the device of ``xyz0``
                (or canonical data) is used.

        Returns:
            posed_xyz: (`N`, 3) Gaussian centres after applying LBS.
            posed_quat: (`N`, 4) Gaussian rotations after LBS (quaternions).
        """

        if xyz0.ndim != 2 or xyz0.shape[1] != 3:
            raise ValueError("xyz0 must have shape (N, 3)")
        if motions_t.ndim != 2 or motions_t.shape[1] != 3:
            raise ValueError("motions_t must have shape (B, 3)")
        if quat0 is not None and (quat0.ndim != 2 or quat0.shape[1] != 4):
            raise ValueError("quat0 must have shape (N, 4)")

        device = device or xyz0.device

        bones0 = self.bones0.to(device)
        relations = self.relations.to(device)
        skin_indices = self.skin_indices.to(device)
        skin_weights = self.skin_weights.to(device)
        xyz0 = xyz0.to(device)
        motions_t = motions_t.to(device)
        quat0_device = quat0.to(device) if quat0 is not None else None

        posed_xyz, posed_rot, _ = interpolate_motions_speedup(
            bones=bones0,
            motions=motions_t,
            relations=relations,
            weights=skin_weights,
            weights_indices=skin_indices,
            xyz=xyz0,
            quat=quat0_device,
        )

        return posed_xyz, posed_rot
