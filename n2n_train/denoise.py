"""Apply a trained model to a full image (raw ADU in, raw ADU out)."""
import torch
import torch.nn.functional as F

from .unet import UNet


def load_model(checkpoint, device="cpu"):
    ckpt = torch.load(checkpoint, map_location=device)
    cfg = ckpt["config"]
    model = UNet(base=cfg["base"], depth=cfg["depth"]).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval()


@torch.no_grad()
def denoise(model, image):
    """`image`: 2D numpy array or tensor in raw ADU (sum of k frames, as in training)."""
    device = next(model.parameters()).device
    x = torch.as_tensor(image, dtype=torch.float32, device=device)[None, None]
    s = x.mean()
    mult = 2 ** len(model.up)
    h, w = x.shape[-2:]
    x = F.pad(x / s, (0, -w % mult, 0, -h % mult), mode="reflect")
    return (model(x)[..., :h, :w] * s)[0, 0].cpu().numpy()
