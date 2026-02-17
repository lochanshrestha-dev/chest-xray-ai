import os
import json
import torch
import numpy as np
from PIL import Image
import torchxrayvision as xrv
from pathlib import Path

# Define the path to the images
image_dir = os.path.expanduser('~/Desktop/CXR_AI_Project/Data/')

def load_and_predict(image_path, model):
    """
    Load a PNG or JPEG image and predict pathologies
    
    Args:
        image_path: Path to the image file
        model: Pre-trained TorchXRayVision model
    
    Returns:
        Dictionary with pathology names and probabilities, or None if error
    """
    try:
        # Load the image in grayscale
        img = Image.open(image_path).convert('L')
        
        # Convert to numpy array
        img_array = np.array(img).astype(np.float32)
        
        # Normalize to 0-1 range
        img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8)
        
        # Resize to model input size (224x224)
        img = Image.fromarray((img_array * 255).astype(np.uint8))
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).float()
        
        # Make prediction
        with torch.no_grad():
            outputs = model(img_tensor)
        
        # Get probabilities
        probabilities = outputs[0].cpu().numpy()
        
        # Create results dictionary
        results = {}
        for pathology, prob in zip(model.pathologies, probabilities):
            results[pathology] = float(prob)
        
        return results
    
    except Exception as e:
        print(f"  Error processing {image_path}: {str(e)}")
        return None

def process_images(image_dir):
    """Process all PNG and JPEG files in the directory"""
    
    print("Loading pre-trained model...")
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()
    
    # Find all PNG and JPEG files
    image_path = Path(image_dir)
    image_files = list(image_path.glob("*.png")) + list(image_path.glob("*.PNG")) + \
                  list(image_path.glob("*.jpg")) + list(image_path.glob("*.JPG")) + \
                  list(image_path.glob("*.jpeg")) + list(image_path.glob("*.JPEG"))
    
    print(f"Found {len(image_files)} image files\n")
    
    all_results = {}
    
    for idx, image_file in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] Processing: {image_file.name}")
        results = load_and_predict(str(image_file), model)
        
        if results:
            all_results[image_file.name] = results
            # Sort and display top 5 findings
            sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
            print(f"  ✓ Success - Top findings:")
            for pathology, prob in list(sorted_results.items())[:5]:
                print(f"    - {pathology}: {prob*100:.1f}%")
    
    return all_results

if __name__ == '__main__':
    if not os.path.exists(image_dir):
        print(f"Error: Directory not found: {image_dir}")
        print("Please check the path and try again.")
    else:
        print(f"Processing images from: {image_dir}\n")
        results = process_images(image_dir)
        
        # Save results to JSON
        output_file = "xray_predictions.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Processing complete!")
        print(f"Results saved to: {output_file}")
        print(f"Total images processed: {len(results)}")
        print(f"{'='*60}")