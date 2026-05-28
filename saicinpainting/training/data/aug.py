from albumentations.core.transforms_interface import DualTransform, to_tuple
import imgaug.augmenters as iaa
import numpy as np


def _apply_iaa(img, augmenter):
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    result = augmenter.augment_image(img)
    if img.ndim == 2:
        result = result[:, :, 0]
    return result


class IAAAffine2(DualTransform):
    def __init__(
        self,
        scale=(0.7, 1.3),
        translate_percent=None,
        translate_px=None,
        rotate=0.0,
        shear=(-0.1, 0.1),
        order=1,
        cval=0,
        mode="reflect",
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.scale = scale
        self.translate_percent = to_tuple(translate_percent, 0)
        self.translate_px = to_tuple(translate_px, 0)
        self.rotate = to_tuple(rotate)
        self.shear = shear
        self.order = order
        self.cval = cval
        self.mode = mode

    def apply(self, img, **params):
        aug = iaa.Affine(
            scale=dict(x=self.scale, y=self.scale),
            translate_percent=self.translate_percent,
            translate_px=self.translate_px,
            rotate=self.rotate,
            shear=dict(x=self.shear, y=self.shear),
            order=self.order,
            cval=self.cval,
            mode=self.mode,
        )
        return _apply_iaa(img, aug)

    def apply_to_mask(self, mask, **params):
        aug = iaa.Affine(
            scale=dict(x=self.scale, y=self.scale),
            translate_percent=self.translate_percent,
            translate_px=self.translate_px,
            rotate=self.rotate,
            shear=dict(x=self.shear, y=self.shear),
            order=0,
            cval=self.cval,
            mode=self.mode,
        )
        return _apply_iaa(mask, aug)

    def get_transform_init_args_names(self):
        return ("scale", "translate_percent", "translate_px", "rotate", "shear", "order", "cval", "mode")


class IAAPerspective2(DualTransform):
    def __init__(self, scale=(0.05, 0.1), keep_size=True, always_apply=False, p=0.5,
                 order=1, cval=0, mode="replicate"):
        super().__init__(always_apply=always_apply, p=p)
        self.scale = to_tuple(scale, 1.0)
        self.keep_size = keep_size
        self.cval = cval
        self.mode = mode

    def apply(self, img, **params):
        aug = iaa.PerspectiveTransform(scale=self.scale, keep_size=self.keep_size,
                                       mode=self.mode, cval=self.cval)
        return _apply_iaa(img, aug)

    def apply_to_mask(self, mask, **params):
        aug = iaa.PerspectiveTransform(scale=self.scale, keep_size=self.keep_size,
                                       mode=self.mode, cval=self.cval)
        return _apply_iaa(mask, aug)

    def get_transform_init_args_names(self):
        return ("scale", "keep_size")
