from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import random
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
try:
    from torchvision.transforms.functional import InterpolationMode
    has_interpolation_mode = True
except ImportError:
    has_interpolation_mode = False

if has_interpolation_mode:
    _torch_interpolation_to_str = {
        InterpolationMode.NEAREST: 'nearest',
        InterpolationMode.BILINEAR: 'bilinear',
        InterpolationMode.BICUBIC: 'bicubic',
        InterpolationMode.BOX: 'box',
        InterpolationMode.HAMMING: 'hamming',
        InterpolationMode.LANCZOS: 'lanczos',
    }
    _str_to_torch_interpolation = {b: a for a, b in _torch_interpolation_to_str.items()}
else:
    _pil_interpolation_to_torch = {}
    _torch_interpolation_to_str = {}

if hasattr(Image, "Resampling"):
    _pil_interpolation_to_str = {
        Image.Resampling.NEAREST: 'nearest',
        Image.Resampling.BILINEAR: 'bilinear',
        Image.Resampling.BICUBIC: 'bicubic',
        Image.Resampling.BOX: 'box',
        Image.Resampling.HAMMING: 'hamming',
        Image.Resampling.LANCZOS: 'lanczos',
    }
else:
    _pil_interpolation_to_str = {
        Image.NEAREST: 'nearest',
        Image.BILINEAR: 'bilinear',
        Image.BICUBIC: 'bicubic',
        Image.BOX: 'box',
        Image.HAMMING: 'hamming',
        Image.LANCZOS: 'lanczos',
    }

_str_to_pil_interpolation = {b: a for a, b in _pil_interpolation_to_str.items()}


def str_to_interp_mode(mode_str):
    if has_interpolation_mode:
        return _str_to_torch_interpolation[mode_str]
    else:
        return _str_to_pil_interpolation[mode_str]


_RANDOM_INTERPOLATION = (str_to_interp_mode('bilinear'), str_to_interp_mode('bicubic'))


class ResizeKeepRatio(transforms.Resize):
    def __init__(self, target_size, interpolation='bicubic', fill=0):
        super().__init__(target_size)
        self.target_size = target_size  # (h, w)
        self.fill = fill
        if interpolation == 'random':
            self.interpolation = _RANDOM_INTERPOLATION
        else:
            self.interpolation = str_to_interp_mode(interpolation)

    def __call__(self, img):
        original_w, original_h = img.size
        target_h, target_w = self.target_size

        # 计算缩放比例
        ratio = min(target_w / original_w, target_h / original_h)
        new_w = int(original_w * ratio)
        new_h = int(original_h * ratio)

        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation

        # 缩放
        img_resized = F.resize(img, (new_h, new_w), interpolation)

        pad_right = target_w - new_w
        pad_bottom = target_h - new_h

        img_padded = F.pad(
            img_resized,
            (0, 0, pad_right, pad_bottom),
            fill=self.fill
        )
        return img_padded


def get_processor(training, **kwargs):
    input_size = kwargs.get("input_size", 1024)
    color_fill = kwargs.get("color_fill", 0)
    color_jitter = kwargs.get("color_jitter", 0.3)
    train_interpolation = kwargs.get("train_interpolation", "bicubic")
    # this should always dispatch to transforms_imagenet_train
    if training:
        processor = transforms.Compose([
            ResizeKeepRatio(
                target_size=(input_size, input_size),
                interpolation=train_interpolation,
                fill=color_fill
            ),
            transforms.RandomApply(
                [transforms.ColorJitter(
                    brightness=color_jitter,
                    contrast=color_jitter,
                    saturation=color_jitter,
                )],
                p=0.3
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])
    else:
        processor = transforms.Compose([
            ResizeKeepRatio(target_size=(input_size, input_size), fill=color_fill),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])
    return processor
