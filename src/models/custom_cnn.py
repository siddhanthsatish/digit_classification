"""
Custom CNN architecture trained from scratch for SVHN digit classification.
Output size is ``num_classes`` (default 11): digits 0–9 (indices 0–9) + background (index 10).

Conv width and dropout are driven by ``config.json`` keys under the ``custom`` section:
``filters``, ``dropout_conv``, ``dropout_fc``, ``fc_hidden``.
"""

import torch.nn as nn

CUSTOM_CNN_CFG_KEYS = ("filters", "dropout_conv", "dropout_fc", "fc_hidden")


def custom_cnn_kwargs_from_cfg(custom_cfg):
    """Map the ``custom`` subsection of config.json to ``CustomCNN`` constructor kwargs."""
    return {k: custom_cfg[k] for k in CUSTOM_CNN_CFG_KEYS if k in custom_cfg}


class CustomCNN(nn.Module):
    """
    Lightweight CNN trained from scratch on 32×32 RGB crops.

    Three conv blocks (each: two 3×3 convs → BN → ReLU → max-pool → spatial dropout).
    Flattened features pass through one hidden FC layer (``fc_hidden`` units, dropout ``dropout_fc``),
    then logits for ``num_classes``.
    """

    def __init__(
        self,
        num_classes=11,
        filters=(32, 64, 128),
        dropout_conv=0.25,
        dropout_fc=0.5,
        fc_hidden=512,
    ):
        super().__init__()
        if len(filters) != 3:
            raise ValueError("filters must contain exactly 3 ints (one per conv block)")
        f1, f2, f3 = (int(x) for x in filters)

        layers = []
        cin = 3
        for fout in (f1, f2, f3):
            layers.extend([
                nn.Conv2d(cin, fout, kernel_size=3, padding=1),
                nn.BatchNorm2d(fout),
                nn.ReLU(inplace=True),
                nn.Conv2d(fout, fout, kernel_size=3, padding=1),
                nn.BatchNorm2d(fout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(dropout_conv),
            ])
            cin = fout

        self.features = nn.Sequential(*layers)
        flat_dim = f3 * 4 * 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_fc),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
