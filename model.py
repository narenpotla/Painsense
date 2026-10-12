"""
PainSense - model definitions.

CNNEncoder:
    ResNet18 backbone (ImageNet-pretrained) -> embedding.

CNNClassifier:
    CNNEncoder + classification head -> NoPain / Pain.
    Used for stage-1 frame classification.

PainCNNLSTM:
    CNNEncoder applied to every frame
    -> LSTM over real video sequences
    -> Pain / NoPain classification for each timestep.
"""

import torch
import torch.nn as nn
import torchvision.models as tvm

import config


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


class CNNClassifier(nn.Module):
    """
    Stage 1:

    Individual face frame
        ↓
    CNN
        ↓
    NoPain / Pain

    Classes:
        0 = NoPain
        1 = Pain
    """

    def __init__(
        self,
        embed_dim=config.EMBED_DIM
    ):
        super().__init__()

        self.encoder = CNNEncoder(
            embed_dim=embed_dim
        )

        # Two output logits:
        # 0 = NoPain
        # 1 = Pain
        self.head = nn.Linear(
            embed_dim,
            2
        )

    def forward(self, x):

        emb = self.encoder(x)

        return self.head(emb)


class PainCNNLSTM(nn.Module):
    """
    Stage 2:

    Real consecutive video frames
        ↓
    CNN / ResNet18
        ↓
    Frame-level spatial features
        ↓
    LSTM
        ↓
    Pain / NoPain logits

    Input:
        (B, T, C, H, W)

    Output:
        (B, T, 2)

    Classes:
        0 = NoPain
        1 = Pain
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

        # Two output logits for each timestep.
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        """
        x shape:

            (B, T, C, H, W)

        Example:

            (8, 8, 3, 224, 224)

        means:
            8 sequences
            8 frames per sequence
            RGB
            224 x 224
        """

        b, t, c, h, w = x.shape

        # Treat every frame as an individual image
        # while passing it through the shared CNN.
        x = x.reshape(
            b * t,
            c,
            h,
            w
        )

        # CNN spatial features.
        emb = self.encoder(x)

        # Put the time dimension back.
        emb = emb.reshape(
            b,
            t,
            -1
        )

        # Temporal processing.
        lstm_out, _ = self.lstm(emb)

        # Classification for every timestep.
        logits = self.head(lstm_out)

        # (B, T, 2)
        return logits

    def load_cnn_weights(
        self,
        cnn_classifier_state_dict
    ):
        """
        Transfer the trained CNN encoder from
        CNNClassifier into the CNN-LSTM.
        """

        encoder_state = {
            k.replace("encoder.", ""): v
            for k, v in cnn_classifier_state_dict.items()
            if k.startswith("encoder.")
        }

        self.encoder.load_state_dict(
            encoder_state
        )


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

    def load_cnn_weights(self, cnn_classifier_state_dict):
        encoder_state = {
            k.replace("encoder.", ""): v
            for k, v in cnn_classifier_state_dict.items()
            if k.startswith("encoder.")
        }
        self.encoder.load_state_dict(encoder_state)


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


def imagenet_backbone():
    """Frozen ImageNet ResNet18 returning (N, 512) pooled features."""

    backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Identity()
    backbone.eval()

    for p in backbone.parameters():
        p.requires_grad = False

    return backbone
