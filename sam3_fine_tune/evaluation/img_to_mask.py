#saves masks and overlay 
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import numpy as np


def load_model(bpe_path: str, sam3_checkpoint: Path, finetuned_checkpoint: Path, 
               device: torch.device = None) -> torch.nn.Module:
    """Load SAM3 model with fine-tuned weights.
     Args:
         bpe_path: Path to the BPE tokenizer file.
         sam3_checkpoint: Path to the original SAM3 model checkpoint.
         finetuned_checkpoint: Path to the fine-tuned model checkpoint.
         device: Optional torch device to load the model on. If None, it will automatically use CUDA if available.
     Returns:
         Loaded SAM3 model with fine-tuned weights."""
        
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = build_sam3_image_model(bpe_path=bpe_path, checkpoint_path=sam3_checkpoint)
    finetuned_weights = torch.load(finetuned_checkpoint, map_location=device)
    model.load_state_dict(finetuned_weights['model'])
    model.eval()
    
    print("Model loaded")
    return model


def infer_on_image(model: torch.nn.Module, image_path: str, prompt: str):
    """Run inference on a single image with a text prompt.
     Args:
         model: The loaded SAM3 model.
         image_path: Path to the input image.
         prompt: Text prompt to guide the segmentation.
     Returns:
         The predicted mask as a torch.Tensor."""
    
    image = Image.open(image_path).convert("RGB")
    processor = Sam3Processor(model, confidence_threshold=0.5)
    inference_state = processor.set_image(image)
    processor.reset_all_prompts(inference_state)
    inference_state = processor.set_text_prompt(state=inference_state, prompt=prompt)
    inference_state['masks'] = (inference_state['masks'] > 0.5).float() * 255
    return inference_state['masks']


def save_mask(mask: torch.Tensor, output_path: str, image_path: str = None):
    """Save the predicted mask as an image file.
     Args:
         mask: The predicted mask as a torch.Tensor.
         output_path: Path to save the output mask image.
         image_path: Optional path to the original image, used to determine size if no masks are detected.
     """    
    
    if mask.dim() == 4:
        if mask.shape[0] == 0:
            # No masks detected, save a black image in original size
            if image_path is not None:
                orig = Image.open(image_path)
                w, h = orig.size
            else:
                h, w = 256, 256
            mask_2d = np.zeros((h, w), dtype='uint8')
        else:
            combined = mask.max(dim=0).values  
            mask_2d = combined.squeeze().cpu().numpy().astype('uint8')  
    else:
        mask_2d = mask.squeeze().cpu().numpy().astype('uint8')
    mask_image = Image.fromarray(mask_2d)
    mask_image.save(output_path)


def save_overlay(image_path: str, mask: torch.Tensor, output_path: str, color=(255, 0, 0), alpha=0.4):
    """Overlay combined mask on the original image and save.
    
    Args:
        image_path: path to the original image
        mask: tensor [N, 1, H, W] with values 0 or 255
        output_path: path to save the overlay image
        color: RGB color for the mask overlay (default: red)
        alpha: transparency of the mask overlay (0=invisible, 1=opaque)
    """
    # Combine all masks via union
    if mask.dim() == 4:
        if mask.shape[0] == 0:
            # No masks detected, create a black mask in original size
            original = Image.open(image_path).convert('RGB')
            w, h = original.size
            mask_2d = np.zeros((h, w), dtype='uint8')
        else:
            combined = mask.max(dim=0).values  # [1, H, W]
            mask_2d = combined.squeeze().cpu().numpy().astype('uint8') 
            original = Image.open(image_path).convert('RGB')
    else:
        mask_2d = mask.squeeze().cpu().numpy().astype('uint8')  
        original = Image.open(image_path).convert('RGB')

    # Load and resize original image to match mask size
    h, w = mask_2d.shape
    original_resized = original.resize((w, h), Image.BILINEAR)
    img_np = np.array(original_resized, dtype=np.float32)

    # Create colored overlay where mask is active
    overlay = np.zeros_like(img_np)
    overlay[mask_2d == 255] = color

    # Blend: only tint masked pixels
    mask_bool = mask_2d == 255
    img_np[mask_bool] = (1 - alpha) * img_np[mask_bool] + alpha * overlay[mask_bool]

    overlay_image = Image.fromarray(img_np.astype('uint8'))
    overlay_image.save(output_path)



def batch_save_masks(img_dir: Path, model, output_dir: Path):
    """Process all images in a directory, save masks and overlays.
     Args:
         img_dir: Directory containing input images.
         model: The loaded SAM3 model.
         output_dir: Directory to save output masks and overlays.
         prompt: Text prompt to guide the segmentation."""
    
    masks_dir = output_dir / 'masks'
    overlays_dir = output_dir / 'overlays'
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(list(img_dir.glob('*.png')))
    for img_path in tqdm(image_files, desc="Processing images"):
        file_name = img_path.stem
        masks = infer_on_image(model, str(img_path), prompt="et_pin")
        save_mask(masks, str(masks_dir / f'{file_name}_mask.png'), image_path=str(img_path))
        save_overlay(str(img_path), masks, str(overlays_dir / f'{file_name}_overlay.png'))


def main():
    checkpoint_path = Path("/home/hpc_storage/data/projects/userA/sam3_data/experiments/testing/checkpoints/checkpoint_10.pt")
    bpe_path = '/home/hpc_storage/data/projects/userA/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz'
    sam3_checkpoint = Path('/home/hpc_storage/data/projects/userA/sam3/models--facebook--sam3/sam3.pt')
    images_dir = Path('/home/hpc_storage/data/projects/userA/sam3_data/multi_test/images/et_pin')
    output_dir = Path('/home/hpc_storage/data/projects/userA/sam3_data/experiments/testing/masks_and_overlays')
    
    model = load_model(bpe_path, sam3_checkpoint, checkpoint_path)
    batch_save_masks(images_dir, model, output_dir)


if __name__ == "__main__":
    main()


