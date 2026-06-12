import cv2
import torch

from midas.midas_net_custom import MidasNet_small
from midas.transforms import Resize, NormalizeImage, PrepareForNet

from torchvision.transforms import Compose


MODEL_TYPE = "midas_v21_small_256"
DEFAULT_MODEL_PATH = "../weights/midas_v21_small_256.pt"


def load_model(device, model_path, optimize=False, height=None, square=False):
    """Load the MiDaS v2.1 small Torch model.

    This is the MiDaS v2.1 small branch from the author's model loader,
    reduced to the only model used by this handoff package.
    """
    keep_aspect_ratio = not square

    model = MidasNet_small(
        model_path,
        features=64,
        backbone="efficientnet_lite3",
        exportable=True,
        non_negative=True,
        blocks={"expand": True},
    )
    net_w, net_h = 256, 256
    resize_mode = "upper_bound"
    normalization = NormalizeImage(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    print("Model loaded, number of parameters = {:.0f}M".format(sum(p.numel() for p in model.parameters()) / 1e6))

    if height is not None:
        net_w, net_h = height, height

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=keep_aspect_ratio,
                ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    model.eval()

    if optimize and (device == torch.device("cuda")):
        model = model.to(memory_format=torch.channels_last)
        model = model.half()

    model.to(device)

    return model, transform, net_w, net_h
