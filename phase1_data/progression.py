# progression.py
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import torchvision.transforms.v2 as T
import cv2
import random
import logging

logger = logging.getLogger(__name__)

class DiseaseProgressionSimulator:
    """
    Simulate visual effects of disease progression stages for cotton leaves.
    This is primarily an augmentation technique.
    """

    def __init__(self):
        self.stage_effects = {
            'early': T.Compose([
                T.ColorJitter(brightness=(0.9, 1.1), contrast=(0.9, 1.1)),
            ]),
            'mid': T.Compose([
                T.ColorJitter(brightness=(0.7, 1.3), contrast=(0.7, 1.3), saturation=(0.8, 1.2)),
                T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.5),
                T.RandomApply([self._add_small_lesions_pil], p=0.3),
            ]),
            'advanced': T.Compose([
                T.ColorJitter(brightness=(0.5, 1.0), contrast=(0.5, 1.0), saturation=(0.5, 1.0), hue=(-0.1, 0.1)),
                T.RandomApply([T.GaussianBlur(kernel_size=(3, 5), sigma=(0.5, 2.5))], p=0.7),
                T.RandomApply([self._add_lesions_cv], p=0.6),
                T.RandomErasing(p=0.2, scale=(0.01, 0.05), ratio=(0.3, 3.3), value='random'),
            ])
            # 'unknown' stage will pass through without changes by default
        }
        logger.info("DiseaseProgressionSimulator initialized with stage effects.")

    def _add_lesions_cv(self, img: Image.Image) -> Image.Image:
        """Simulate lesions by adding dark spots using OpenCV for more control."""
        try:
            # Convert PIL Image to OpenCV format (NumPy array, BGR)
            img_cv_bgr = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR)
            h, w = img_cv_bgr.shape[:2]
            num_lesions = random.randint(1, 5) # Adjusted range for visual clarity

            for _ in range(num_lesions):
                x, y = random.randint(0, w - 1), random.randint(0, h - 1)
                radius = random.randint(int(min(h, w) * 0.01), int(min(h, w) * 0.06)) # Adjusted size

                # Define lesion color (B, G, R format for OpenCV)
                # Ensure these are Python integers
                lesion_b = int(random.uniform(10, 40))
                lesion_g = int(random.uniform(20, 50))
                lesion_r = int(random.uniform(30, 60))
                lesion_color_tuple = (lesion_b, lesion_g, lesion_r)

                cv2.circle(img_cv_bgr, (x, y), radius, lesion_color_tuple, -1) # -1 for filled circle

            # Convert back to PIL Image (from BGR to RGB)
            return Image.fromarray(cv2.cvtColor(img_cv_bgr, cv2.COLOR_BGR2RGB))
        except Exception as e:
            logger.error(f"Error in _add_lesions_cv: {e}", exc_info=True)
            return img # Return original image on error

    def _add_small_lesions_pil(self, img: Image.Image) -> Image.Image:
        """Simulate smaller, perhaps more subtle lesions using PIL drawing."""
        try:
            # For better lesions, OpenCV (_add_lesions_cv) is generally more flexible.
            # This is a simpler PIL alternative.
            from PIL import ImageDraw # Import here to keep cv2 dependency optional if this is preferred
            
            # Work on a copy
            img_copy = img.copy()
            draw = ImageDraw.Draw(img_copy)
            num_lesions = random.randint(1, 3)
            
            for _ in range(num_lesions):
                x, y = random.randint(0, img_copy.width -1), random.randint(0, img_copy.height -1)
                radius = random.randint(max(1, int(min(img_copy.width, img_copy.height) * 0.005)), \
                                    max(3, int(min(img_copy.width, img_copy.height) * 0.02)))
                # Darkish, desaturated color
                r_col = random.randint(40,80)
                g_col = random.randint(30,70)
                b_col = random.randint(20,60)
                color = (r_col, g_col, b_col)
                draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)
            return img_copy
        except Exception as e:
            logger.error(f"Error in _add_small_lesions_pil: {e}", exc_info=True)
            return img


    def apply(self, img: Image.Image, stage: str) -> Image.Image:
        """
        Apply stage-specific transformations.
        Args:
            img (Image.Image): Input PIL Image.
            stage (str): Disease stage ('early', 'mid', 'advanced', or other defined).
        Returns:
            Image.Image: Transformed PIL Image.
        """
        if stage not in self.stage_effects:
            # logger.debug(f"Stage '{stage}' not found in simulator effects. Returning original image.")
            return img

        try:
            # Ensure working with a copy if transforms are in-place (most torchvision are not for PIL)
            # but custom functions might be.
            return self.stage_effects[stage](img.copy()) # Apply to a copy
        except Exception as e:
            logger.error(f"Error applying progression for stage '{stage}': {e}", exc_info=True)
            return img # Return original image on error