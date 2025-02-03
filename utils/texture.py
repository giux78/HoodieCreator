from PIL import Image, ImageOps
import numpy as np

def apply_texture(base_image_path, texture_image_path, output_path, opacity=0.5):
    """
    Apply a texture image to a base image.
    
    Parameters:
    base_image_path (str): Path to the base image
    texture_image_path (str): Path to the texture image
    output_path (str): Path where the resulting image will be saved
    opacity (float): Opacity of the texture (0.0 to 1.0)
    """
    # Open both images
    base_image = Image.open(base_image_path).convert('RGBA')
    texture_image = Image.open(texture_image_path).convert('RGBA')
    
    # Resize texture to match base image size
    texture_image = texture_image.resize(base_image.size, Image.Resampling.LANCZOS)
    
    # Convert images to numpy arrays
    base_array = np.array(base_image)
    texture_array = np.array(texture_image)
    
    # Create the blended image
    blended_array = np.zeros_like(base_array)
    
    # Blend the images
    for i in range(3):  # RGB channels
        blended_array[:,:,i] = (
            base_array[:,:,i] * (1 - opacity) +
            texture_array[:,:,i] * opacity
        ).astype(np.uint8)
    
    # Preserve the alpha channel from the base image
    blended_array[:,:,3] = base_array[:,:,3]
    
    # Convert back to PIL Image and save
    result_image = Image.fromarray(blended_array)
    result_image.save(output_path)
    return result_image

def apply_texture_multiply(base_image_path, texture_image_path, output_path=None, strength=0.5):
    """
    Apply a texture using multiply blend mode.
    
    Parameters:
    base_image_path (str): Path to the base image
    texture_image_path (str): Path to the texture image
    output_path (str): Path where the resulting image will be saved
    strength (float): Strength of the texture effect (0.0 to 1.0)
    """
    # Open both images
    base_image = Image.open(base_image_path).convert('RGBA')
    texture_image = Image.open(texture_image_path).convert('RGBA')
    
    # Resize texture to match base image size
    texture_image = texture_image.resize(base_image.size, Image.Resampling.LANCZOS)
    
    # Convert to numpy arrays
    base_array = np.array(base_image).astype(float)
    texture_array = np.array(texture_image).astype(float)
    
    # Normalize arrays to 0-1 range
    base_array = base_array / 255
    texture_array = texture_array / 255
    
    # Apply multiply blend
    multiplied = base_array * texture_array
    
    # Blend with original based on strength
    result_array = (base_array * (1 - strength) + multiplied * strength) * 255
    
    # Convert back to uint8
    result_array = result_array.clip(0, 255).astype(np.uint8)
    
    # Convert back to PIL Image and save
    result_image = Image.fromarray(result_array)
    #result_image.save(output_path)
    return result_image