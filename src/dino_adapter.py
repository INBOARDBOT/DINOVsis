import torch
import torch.nn as nn

class DinoFeatureExtractor(nn.Module):
    def __init__(self, segmentor):
        super().__init__()
        self.segmentor = segmentor
        
        for param in self.segmentor.parameters():
            param.requires_grad = False
            
        self.target_layers = [2, 5, 8, 11]

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. Patch Embedding
        x = self.segmentor.patch_embed(x)
        # Your DINOv3 returns [B, Hp, Wp, D] — spatial-first layout
        
        patch_size = self.segmentor.patch_embed.proj.kernel_size
        Hp = H // patch_size[0]   # 896 // 16 = 56
        Wp = W // patch_size[1]

        # Normalise to [B, Hp*Wp, D] regardless of patch_embed layout
        if x.ndim == 4 and x.shape[-1] != x.shape[1]:
            # [B, Hp, Wp, D] → [B, Hp*Wp, D]
            x = x.reshape(B, Hp * Wp, -1)
        elif x.ndim == 4:
            # [B, D, Hp, Wp] → [B, Hp*Wp, D]
            x = x.flatten(2).transpose(1, 2)
        # ndim == 3: already [B, Hp*Wp, D], do nothing

        # 2. Prepend special tokens
        cls_tokens  = self.segmentor.cls_token.expand(B, -1, -1)   # [B, 1, D]
        mask_tokens = self.segmentor.mask_token.expand(B, -1, -1)  # [B, 1, D]

        storage_tokens = self.segmentor.storage_tokens
        if storage_tokens.ndim == 2:
            storage_tokens = storage_tokens.unsqueeze(0)            # [1, N, D]
        storage_tokens = storage_tokens.expand(B, -1, -1)           # [B, N, D]

        # All tensors now share last dim D=384 — cat is safe
        x = torch.cat((cls_tokens, mask_tokens, storage_tokens, x), dim=1)

        # 3. Pass through transformer blocks, collect features
        features = []
        num_special_tokens = (
            cls_tokens.shape[1]
            + mask_tokens.shape[1]
            + storage_tokens.shape[1]
        )

        for i, block in enumerate(self.segmentor.blocks):
            x = block(x)

            if i in self.target_layers:
                patch_tokens = x[:, num_special_tokens:, :]  # [B, Hp*Wp, D]
                feat_2d = (
                    patch_tokens
                    .transpose(1, 2)        # [B, D, Hp*Wp]
                    .reshape(B, -1, Hp, Wp) # [B, D, Hp, Wp]
                    .contiguous()
                )
                features.append(feat_2d)

        return features  # 4 × [B, 384, 56, 56]