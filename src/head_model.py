# head_model.py
import torch
import torch.nn as nn
from dino_adapter import DinoFeatureExtractor




class CoxRiskNet(nn.Module):
    """
    Risk-score head on top of a frozen DINOv3 backbone, with feature-level
    mean pooling across a patient's slices.

    A patient's MRI volume is sliced into multiple 2-D images. Each slice is
    pushed through the backbone independently to get its own feature map,
    those feature maps are spatially pooled to per-slice embeddings, and
    THOSE embeddings are averaged across all slices into a single
    per-patient embedding before the MLP head produces one risk score. This
    guarantees exactly one risk score per patient regardless of how many
    slices their volume has — which is required for the Cox partial
    likelihood (cox_loss), since it compares risk *between patients*, not
    between slices.

    Parameters
    ----------
    segmentor   : the DINOv3 backbone returned by load_segmentor()
    embed_dim   : feature dimension of the chosen ViT backbone
                  vits16 / vits16plus → 384
                  vitb16              → 768
                  vitl16              → 1024
                  vith16plus          → 1280
    num_classes : unused for the Cox head (kept for call-site compatibility
                  with the segmentation head's constructor signature).
    """

    def __init__(self, segmentor, num_classes: int = None, embed_dim: int = 384):
        super().__init__()

        self.extractor = DinoFeatureExtractor(segmentor)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.net = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),   # single risk score per patient
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (num_slices, 3, H, W) — all slices belonging to ONE patient.

        Returns a single scalar risk score, shape (1,) (or shape () if you
        squeeze further), NOT one score per slice.
        """
        # Use the deepest feature map (last entry) from the backbone.
        features = self.extractor(x)          # [num_slices, D, Hp, Wp]
        feat = features[-1]
        pooled_per_slice = self.pool(feat).flatten(1)        # [num_slices, D]

        # Aggregate across slices → single embedding for this patient.
        patient_embedding = pooled_per_slice.mean(dim=0, keepdim=True)  # [1, D]

        risk = self.net(patient_embedding)     # [1, 1]
        return risk.squeeze(-1)                # [1]