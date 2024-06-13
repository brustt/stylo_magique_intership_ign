from typing import Tuple
import numpy as np
import torch.nn as nn
import torch
from magic_pen.config import DEVICE


def generate_bboxes(size: Tuple[int, int]):
    bboxes = np.array([[1, 2, 5, 6], [10, 20, 50, 60]])
    bboxes = np.repeat(bboxes, size[1] // bboxes.shape[0], axis=0)
    arr = np.expand_dims(np.random.randint(0, 100, bboxes.shape[0]), axis=1)
    noise = bboxes + np.repeat(arr, bboxes.shape[1], axis=1)
    # batch_noise = np.expand_dims(noise, axis=0).repeat(size[0], axis=0)
    return noise


class DummyModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.enc = self.build_block_encoder(in_dim, in_dim * 2)
        self.dec = self.build_block_decoder(in_dim * 4, out_dim)

    def forward(self, x):
        x = x["img_A"][:2]  # simulate paire
        x1 = self.enc(x)
        x2 = self.dec(x1)
        return dict(
            masks=x2,
            iou_preds=torch.as_tensor(np.ones((x.shape[0], 1))),
            confidence_scores=torch.as_tensor(np.ones((x.shape[0], 1))),
            bboxes=torch.as_tensor(generate_bboxes(size=(x.shape[0], 10))),
        )

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
