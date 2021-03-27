from .auto_augment import (AutoAugment, BrightnessTransform, ColorTransform,
                           ContrastTransform, EqualizeTransform, Rotate, Shear,
                           Translate)
from .compose import Compose
from .formating import (Collect, DefaultFormatBundle, ImageToTensor,
                        ToDataContainer, ToTensor, Transpose, to_tensor)
from .instaboost import InstaBoost
from .loading import (LoadAnnotations, LoadImageFromFile, LoadImageFromWebcam,
                      LoadMultiChannelImageFromFiles, LoadProposals)
from .test_time_aug import MultiScaleFlipAug
from .transforms import (Albu, CutOut, Expand, MinIoURandomCrop, Normalize,
                         Pad, PhotoMetricDistortion, RandomCenterCropPad,
                         RandomCrop, RandomFlip, Resize, SegRescale,
                         Mixup, ReplaceBackground)  #  clw modify

from .concat import Concat
from .concat_6channel import Concat6
from .concat_template import ConcatTemplate, LoadTemplate
from .loading import LoadMosaicImageAndAnnotations
from .loading_reppointsv2 import (LoadRPDV2Annotations, LoadDenseRPDV2Annotations)  # clw add
from .formating_reppointsv2 import RPDV2FormatBundle

__all__ = [
    'Compose', 'to_tensor', 'ToTensor', 'ImageToTensor', 'ToDataContainer',
    'Transpose', 'Collect', 'DefaultFormatBundle', 'LoadAnnotations',
    'LoadImageFromFile', 'LoadImageFromWebcam',
    'LoadMultiChannelImageFromFiles', 'LoadProposals', 'MultiScaleFlipAug',
    'Resize', 'RandomFlip', 'Pad', 'RandomCrop', 'Normalize', 'SegRescale',
    'MinIoURandomCrop', 'Expand', 'PhotoMetricDistortion', 'Albu',
    'InstaBoost', 'RandomCenterCropPad', 'AutoAugment', 'CutOut', 'Shear',
    'Rotate', 'ColorTransform', 'EqualizeTransform', 'BrightnessTransform',
    'ContrastTransform', 'Translate'
    ,'Concat', 'Concat6', 'LoadMosaicImageAndAnnotations' # clw modify
    ,'ConcatTemplate', 'LoadTemplate', 'Mixup', 'ReplaceBackground'
    , 'LoadRPDV2Annotations', 'LoadDenseRPDV2Annotations', 'RPDV2FormatBundle'
]
