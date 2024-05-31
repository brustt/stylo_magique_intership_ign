import numpy as np
import torch.nn as nn
import torch
from magic_pen.config import DEVICE


class DummyModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.enc = self.build_block_encoder(in_dim, in_dim * 2)
        self.dec = self.build_block_decoder(in_dim * 4, out_dim)

    def forward(self, x):
        x = x["img_A"]
        x1 = self.enc(x)
        x2 = self.dec(x1)
        return dict(masks=x2, iou_preds=[])

    def build_block_encoder(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
            nn.Conv2d(out_dim, out_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim * 2),
        )

    def build_block_decoder(self, in_dim, out_dim):
        return nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
        )


if __name__ == "__main__":
    img = torch.as_tensor(
        np.random.random((1, 3, 1024, 1024)), dtype=torch.float, device=DEVICE
    )
    model = DummyModel(in_dim=3, out_dim=1)
    pred = model(img)
    print(pred.shape)
