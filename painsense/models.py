"""
painsense.models - network definitions and checkpoint loading.

    CNNEncoder              ResNet18 backbone + linear projection
    PainIntensityLSTM       full CNN-LSTM on images (train_full_model.py);
                            backbone BatchNorm always in eval mode
    FeatureIntensityModel   head on cached 512-d ResNet features, causal:
                            "frame" (no context), "pool" (running mean),
                            "lstm" (LSTM over the window)
    FeatureEnsemble         average of several FeatureIntensityModel heads
    imagenet_backbone       frozen ImageNet ResNet18 feature extractor
    load_feature_checkpoint load the live model (backbone + head(s))

Every model outputs (B, T) scores on the synthetic 0-10 morph scale.
"""

import torch
import torch.nn as nn
import torchvision.models as tvm

from painsense import config


class CNNEncoder(nn.Module):
    """
    ResNet18 feature extractor.

    Input:
        (B, C, H, W)

    Output:
        (B, EMBED_DIM)
    """

    def __init__(
        self,
        embed_dim=config.EMBED_DIM,
        pretrained=True
    ):
        super().__init__()

        weights = (
            tvm.ResNet18_Weights.IMAGENET1K_V1
            if pretrained
            else None
        )

        backbone = tvm.resnet18(weights=weights)

        in_features = backbone.fc.in_features

        # Remove ResNet's original classification layer.
        backbone.fc = nn.Identity()

        self.backbone = backbone

        # Project ResNet features to the dimension
        # expected by the LSTM.
        self.proj = nn.Linear(
            in_features,
            embed_dim
        )

    def forward(self, x):
        # x: (B, C, H, W)

        feats = self.backbone(x)

        # (B, in_features)
        return self.proj(feats)

    def freeze_backbone(self, freeze=True):

        for p in self.backbone.parameters():
            p.requires_grad = not freeze


def backbone_bn_eval(module):
    """Put every BatchNorm layer under `module` in eval mode."""

    for m in module.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


class PainIntensityLSTM(nn.Module):
    """
    CNN-LSTM regression head.

    Input:
        (B, T, C, H, W)

    Output:
        (B, T) expression intensity on the synthetic 0-10 morph scale.

    The ResNet backbone is (partly) frozen, so its BatchNorm layers are
    always kept in eval mode: model.train() would otherwise keep
    updating their running statistics from tiny, highly correlated
    batches.
    """

    def __init__(
        self,
        embed_dim=config.EMBED_DIM,
        hidden_dim=config.LSTM_HIDDEN
    ):
        super().__init__()

        self.encoder = CNNEncoder(
            embed_dim=embed_dim
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def train(self, mode=True):
        super().train(mode)
        backbone_bn_eval(self.encoder.backbone)
        return self

    # -- split forward so live inference can cache backbone features --

    def encode_backbone(self, x):
        """(N, C, H, W) -> (N, 512) pooled backbone features."""
        return self.encoder.backbone(x)

    def score_features(self, feats):
        """(B, T, 512) cached backbone features -> (B, T) scores."""
        emb = self.encoder.proj(feats)
        lstm_out, _ = self.lstm(emb)
        return self.head(lstm_out).squeeze(-1)

    def forward(self, x):
        b, t, c, h, w = x.shape

        x = x.reshape(b * t, c, h, w)
        feats = self.encode_backbone(x)
        feats = feats.reshape(b, t, -1)
        return self.score_features(feats)


class FeatureIntensityModel(nn.Module):
    """
    Regression head on cached, frozen ImageNet ResNet18 features.

    Input:
        (B, T, FEATURE_DIM) features, oldest frame first.

    Output:
        (B, T) scores; output t only depends on frames <= t (causal).

    variant:
        "frame" - each frame scored on its own (no temporal context)
        "pool"  - running (causal) mean of projected features, then head
        "lstm"  - LSTM over projected features, then head

    Parameter names (proj / lstm / head) match PainIntensityLSTM, so the
    "lstm" variant is the same head as the end of the full model.
    """

    VARIANTS = ("frame", "pool", "lstm")

    def __init__(
        self,
        variant="lstm",
        feature_dim=config.FEATURE_DIM,
        embed_dim=config.EMBED_DIM,
        hidden_dim=config.LSTM_HIDDEN
    ):
        super().__init__()

        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")

        self.variant = variant
        self.proj = nn.Linear(feature_dim, embed_dim)

        if variant == "lstm":
            self.lstm = nn.LSTM(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                batch_first=True
            )
            head_in = hidden_dim
        else:
            head_in = embed_dim

        self.head = nn.Sequential(
            nn.Linear(head_in, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, feats):
        z = self.proj(feats)

        if self.variant == "lstm":
            z, _ = self.lstm(z)
        elif self.variant == "pool":
            steps = torch.arange(
                1, z.shape[1] + 1, device=z.device, dtype=z.dtype
            ).view(1, -1, 1)
            z = z.cumsum(dim=1) / steps

        return self.head(z).squeeze(-1)


def imagenet_backbone(state_dict=None):
    """
    Frozen ImageNet ResNet18 returning (N, 512) pooled features.

    state_dict: backbone weights stored in a checkpoint. When given,
    nothing is downloaded, so the live demo works offline.
    """

    if state_dict is None:
        backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()
    else:
        backbone = tvm.resnet18(weights=None)
        backbone.fc = nn.Identity()
        backbone.load_state_dict(state_dict)

    backbone.eval()

    for p in backbone.parameters():
        p.requires_grad = False

    return backbone


class FeatureEnsemble(nn.Module):
    """Average of several FeatureIntensityModel heads (same input/output)."""

    def __init__(self, heads):
        super().__init__()
        self.heads = nn.ModuleList(heads)

    def forward(self, feats):
        return torch.stack([h(feats) for h in self.heads]).mean(dim=0)


def load_feature_checkpoint(path, device=None):
    """
    Load checkpoints/pain_intensity_features.pt.

    Returns (backbone, head, payload). head is a FeatureIntensityModel or
    a FeatureEnsemble when the checkpoint stores several heads
    ("state_dicts"). Uses the stored backbone weights when present, so
    nothing is downloaded.
    """

    if device is None:
        device = config.DEVICE

    payload = torch.load(path, map_location=device)
    variant = payload.get("variant", "lstm")
    states = payload.get("state_dicts") or [payload["state_dict"]]

    heads = []

    for state in states:
        head = FeatureIntensityModel(variant)
        head.load_state_dict(state)
        heads.append(head)

    head = heads[0] if len(heads) == 1 else FeatureEnsemble(heads)
    head = head.to(device).eval()

    backbone = imagenet_backbone(
        payload.get("backbone_state_dict")
    ).to(device)

    return backbone, head, payload
