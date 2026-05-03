# =============================================================================
# model.py
# ResNet-50 based model for 30-class agricultural crop classification.
#
# Architecture choices:
#   - Backbone: torchvision ResNet-50 (pretrained ImageNet weights)
#   - All layers are fine-tuned (no layer freezing)
#   - Classifier head replaced with:
#       Linear(2048 → 512) → BatchNorm1d → ReLU → Dropout → Linear(512 → 30)
#   - BatchNorm and Dropout are already present in the ResNet backbone;
#     we add extra ones in the custom head for regularisation.
# =============================================================================

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

import config


class CropClassifier(nn.Module):
    """
    ResNet-50 backbone with a custom classification head for num_classes crops.

    Parameters
    ----------
    num_classes  : number of output classes (default: config.num_classes)
    pretrained   : load ImageNet-pretrained weights for backbone (default: True)
    dropout_rate : dropout probability before final linear layer
    """

    def __init__(
        self,
        num_classes:  int   = config.num_classes,
        pretrained:   bool  = True,
        dropout_rate: float = config.dropout_rate,
    ):
        super().__init__()

        # ── Backbone ──────────────────────────────────────────────────────
        weights  = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Remove the original fully-connected head; keep everything up to
        # the global average pooling (output shape: [B, 2048])
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        in_features = backbone.fc.in_features  # 2048 for ResNet-50

        # ── Custom classifier head ────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),                           # [B, 2048, 1, 1] → [B, 2048]
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes),            # logits (no softmax here)
        )

        # Initialise the new linear layers with Kaiming uniform
        self._init_weights()

    # ------------------------------------------------------------------

    def _init_weights(self):
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor of shape [B, 3, H, W]

        Returns
        -------
        logits : torch.Tensor of shape [B, num_classes]
        """
        features = self.feature_extractor(x)   # [B, 2048, 1, 1]
        logits   = self.classifier(features)   # [B, num_classes]
        return logits

    # ------------------------------------------------------------------

    def get_num_params(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Model factory (convenience)
# ---------------------------------------------------------------------------

def build_model(pretrained: bool = True) -> CropClassifier:
    """Instantiates and returns the model. Entry point for other modules."""
    model = CropClassifier(
        num_classes  = config.num_classes,
        pretrained   = pretrained,
        dropout_rate = config.dropout_rate,
    )
    return model


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = build_model(pretrained=False)
    print(model)
    dummy = torch.randn(4, config.input_channels, config.resize_y, config.resize_x)
    out   = model(dummy)
    print(f"Input shape  : {dummy.shape}")
    print(f"Output shape : {out.shape}")
    print(f"Trainable params: {model.get_num_params():,}")
