# models.py
import torch
import torch.nn as nn
import torchvision.models as tv_models
import timm
import logging

logger = logging.getLogger(__name__)

def get_baseline_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Loads a pre-trained baseline model (CNN or ViT) and modifies its classifier.
    Args:
        model_name (str): Name of the model (e.g., "resnet50", "efficientnet_b0", "vit_base_patch16_224").
        num_classes (int): Number of output classes.
        pretrained (bool): Whether to load pre-trained weights.
    Returns:
        nn.Module: The model with a new classifier head.
    """
    logger.info(f"Loading model: {model_name} with pretrained={pretrained} for {num_classes} classes.")
    model = None

    # Try loading from torchvision first
    if hasattr(tv_models, model_name):
        model_builder = tv_models.get_model_builder(model_name)
        try:
            # Modern torchvision (>=0.13) uses a 'weights' parameter
            # For ResNet50, a common set of weights is ResNet50_Weights.IMAGENET1K_V1 or _V2
            # We can generalize or try to fetch default weights
            if pretrained:
                try:
                    # Attempt to get default weights enum if available (e.g., ResNet50_Weights.DEFAULT)
                    weights_enum = tv_models.get_model_weights(model_name)
                    default_weights = weights_enum.DEFAULT if hasattr(weights_enum, 'DEFAULT') else weights_enum.IMAGENET1K_V1
                    model = model_builder(weights=default_weights)
                    logger.info(f"Loaded {model_name} with weights {default_weights} from torchvision.")
                except AttributeError as e: # If get_model_weights or DEFAULT fails
                    logger.warning(f"Could not get default weights for {model_name} via get_model_weights: {e}. Falling back.")
                    # Fallback for models where weights enum might not be structured with DEFAULT or specific name
                    # This might happen with less common models or older torchvision versions still having `weights` param
                    if "weights" in model_builder.__init__.__code__.co_varnames: # Check if __init__ itself takes weights
                         model = model_builder(weights="IMAGENET1K_V1" if pretrained else None)
                         logger.info(f"Loaded {model_name} with 'IMAGENET1K_V1' weights (fallback) from torchvision.")
                    else: # If __init__ does not take weights, try pretrained (older torchvision)
                         model = model_builder(pretrained=pretrained)
                         logger.info(f"Loaded {model_name} with pretrained={pretrained} (older API) from torchvision.")

            else: # Not pretrained
                model = model_builder(weights=None, pretrained=False) # Try both for safety
                logger.info(f"Loaded {model_name} without pretrained weights from torchvision.")

        except Exception as e_tv: # Catch any error during torchvision model instantiation
            logger.warning(f"Failed to load {model_name} from torchvision with modern API: {e_tv}. Will try timm.")
            model = None # Ensure model is None to trigger timm


        if model: # If model was successfully loaded from torchvision, adapt classifier
            if 'resnet' in model_name or 'resnext' in model_name:
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, num_classes)
            elif 'efficientnet' in model_name:
                # EfficientNet in torchvision has `model.classifier` which is a Sequential block
                # The last layer is usually a Linear layer.
                if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential) and \
                   len(model.classifier) > 0 and isinstance(model.classifier[-1], nn.Linear):
                    num_ftrs = model.classifier[-1].in_features
                    model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
                else: # Fallback for other EfficientNet structures or if classifier structure is different
                    logger.warning(f"Could not automatically adapt classifier for torchvision EfficientNet '{model_name}'. Check structure.")
                    model = None # Force timm if adaptation fails
            elif 'densenet' in model_name:
                num_ftrs = model.classifier.in_features
                model.classifier = nn.Linear(num_ftrs, num_classes)
            elif 'mobilenet_v2' in model_name:
                num_ftrs = model.classifier[1].in_features # Last layer in classifier Sequential
                model.classifier[1] = nn.Linear(num_ftrs, num_classes)
            elif 'mobilenet_v3' in model_name:
                num_ftrs = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
            # Add more torchvision models as needed
            else:
                logger.warning(f"Classifier modification for torchvision model '{model_name}' not explicitly handled. Trying timm if model is None.")
                # If we couldn't adapt, set model to None to try timm
                # This 'else' might not be reached if model loaded but wasn't one of the above,
                # so it's better to ensure model is None if any specific handling fails.
                pass # Assume successful load if no specific adaptation was needed and model is not None

    # If not found or handled/failed in torchvision, try timm
    if model is None:
        logger.info(f"Attempting to load model '{model_name}' from timm.")
        try:
            model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
            logger.info(f"Loaded {model_name} from timm with num_classes={num_classes}.")
        except Exception as e_timm:
            logger.error(f"Failed to load model '{model_name}' from torchvision or timm. Last timm error: {e_timm}")
            raise ValueError(f"Unsupported or failed to load model_name: {model_name}")

    return model

if __name__ == '__main__':
    # Test cases
    test_models_to_load = ["resnet50", "vit_base_patch16_224", "efficientnet_b0", "mobilenet_v2"]
    for model_name_test in test_models_to_load:
        try:
            logger.info(f"\n--- Testing {model_name_test} ---")
            m = get_baseline_model(model_name_test, num_classes=10, pretrained=True)
            logger.info(f"{model_name_test} loaded successfully.")
            
            # Determine input size (most timm ViTs need 224, some CNNs are flexible)
            input_size = (224, 224)
            if hasattr(m, 'default_cfg') and 'input_size' in m.default_cfg:
                input_size = m.default_cfg['input_size'][1:] # (H, W)

            dummy_input = torch.randn(2, 3, input_size[0], input_size[1])
            out = m(dummy_input)
            logger.info(f"{model_name_test} output shape: {out.shape}")
            assert out.shape == (2, 10)
        except Exception as e:
            logger.error(f"Error during model loading test for {model_name_test}: {e}", exc_info=True)