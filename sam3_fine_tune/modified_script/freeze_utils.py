import torch.nn as nn
from typing import List
import fnmatch

def freeze_parameters_by_pattern(model: nn.Module, patterns: List[str]):
    """Freeze parameters matching Unix-style patterns."""
    for name, param in model.named_parameters():
        for pattern in patterns:
            if fnmatch.fnmatch(name, pattern):
                param.requires_grad_(False)
                print(f"Frozen: {name}")
                break
    return model

class FreezeEncoderBackboneInitializer:
    """Initializer that freezes encoder and backbone for decoder-only finetuning."""
    
    def __call__(self, model):
        patterns_to_freeze = [
            'backbone.vision_backbone.*',
            'backbone.language_backbone.*',
            'transformer.encoder.*',
        ]
        return freeze_parameters_by_pattern(model, patterns_to_freeze)