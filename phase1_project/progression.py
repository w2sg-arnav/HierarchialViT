# progression.py
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import torchvision.transforms.v2 as T_v2
import cv2
import random
import logging
import torch

logger = logging.getLogger(__name__)

class DiseaseProgressionSimulator:
    """
    Simulate visual effects of disease progression stages for cotton leaves.
    This is primarily an augmentation technique. PIL Images are expected as input and output.
    """
    def __init__(self):
        self.stage_effects = {
            'early': T_v2.Compose([
                T_v2.ColorJitter(brightness=(0.95, 1.05), contrast=(0.95, 1.05)),
            ]),
            'mid': T_v2.Compose([
                T_v2.ColorJitter(brightness=(0.8, 1.1), contrast=(0.8, 1.1), saturation=(0.9, 1.1)),
                T_v2.RandomApply([T_v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.4),
                T_v2.RandomApply([self._add_small_lesions_pil_transform], p=0.3),
            ]),
            'advanced': T_v2.Compose([
                T_v2.ColorJitter(brightness=(0.6, 1.0), contrast=(0.6, 1.0), saturation=(0.7, 1.0), hue=(-0.05, 0.05)),
                T_v2.RandomApply([T_v2.GaussianBlur(kernel_size=(3, 5), sigma=(0.3, 2.0))], p=0.6),
                T_v2.RandomApply([self._add_lesions_cv_transform], p=0.5),
                # RandomErasing needs tensor input, so we apply it if img is converted to tensor, or do it manually
                # For PIL version, we'll skip RandomErasing or implement a PIL version.
            ])
            # 'unknown' stage will pass through without changes by default
        }
        logger.info("DiseaseProgressionSimulator initialized with stage effects.")

    # Wrapper classes for PIL transforms to be used in T_v2.Compose
    class _AddLesionsCVTransform(torch.nn.Module):
        def forward(self, img: Image.Image) -> Image.Image:
            try:
                img_cv_bgr = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR)
                h, w = img_cv_bgr.shape[:2]
                num_lesions = random.randint(1, 5)
                for _ in range(num_lesions):
                    x, y = random.randint(0, w - 1), random.randint(0, h - 1)
                    radius = random.randint(max(1, int(min(h,w)*0.01)), max(2,int(min(h,w)*0.06)))
                    lesion_b = int(random.uniform(10, 40))
                    lesion_g = int(random.uniform(20, 50))
                    lesion_r = int(random.uniform(30, 60))
                    cv2.circle(img_cv_bgr, (x, y), radius, (lesion_b, lesion_g, lesion_r), -1)
                return Image.fromarray(cv2.cvtColor(img_cv_bgr, cv2.COLOR_BGR2RGB))
            except Exception as e:
                logger.error(f"Error in _add_lesions_cv_transform: {e}", exc_info=True)
                return img

    class _AddSmallLesionsPILTransform(torch.nn.Module):
        def forward(self, img: Image.Image) -> Image.Image:
            try:
                from PIL import ImageDraw
                img_copy = img.copy()
                draw = ImageDraw.Draw(img_copy)
                num_lesions = random.randint(1, 3)
                for _ in range(num_lesions):
                    x, y = random.randint(0, img_copy.width-1), random.randint(0, img_copy.height-1)
                    radius = random.randint(max(1,int(min(img_copy.width,img_copy.height)*0.005)),
                                            max(3,int(min(img_copy.width,img_copy.height)*0.02)))
                    r, g, b = random.randint(40,80), random.randint(30,70), random.randint(20,60)
                    draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=(r,g,b))
                return img_copy
            except Exception as e:
                logger.error(f"Error in _add_small_lesions_pil_transform: {e}", exc_info=True)
                return img

    # Make instances of the wrapper classes
    _add_lesions_cv_transform = _AddLesionsCVTransform()
    _add_small_lesions_pil_transform = _AddSmallLesionsPILTransform()

    def apply(self, img: Image.Image, stage: str) -> Image.Image:
        if stage not in self.stage_effects:
            return img
        try:
            # T_v2.Compose expects PIL image and returns PIL image if ToImage() is not in Compose
            return self.stage_effects[stage](img.copy())
        except Exception as e:
            logger.error(f"Error applying progression for stage '{stage}': {e}", exc_info=True)
            return img