import numpy as np
from scipy import ndimage
from skimage.filters import meijering, threshold_otsu
from skimage.morphology import remove_small_objects, closing, disk

# LIBRARIES FOR LOADING, VIEWING, AND SAVING (No OpenCV)
import imageio.v3 as iio
import matplotlib.pyplot as plt

def segment_golgi_dendrite_meijering(axial_slice):
    """
    Creates a binary mask for a dendritic structure from a Golgi-Cox axial slice 
    using the Meijering Neuriteness filter. Assumes dark dendrites on a light background.
    """
    img = np.array(axial_slice, dtype=float)
    
    # 1. Convert to Grayscale if RGB
    if img.ndim == 3:
        img = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
        
    # 2. Invert (Dendrites become bright for the ridge filter)
    img_max = img.max() if img.max() > 0 else 1.0
    inverted = 1.0 - (img / img_max)
    
    # 3. Focus Energy (Laplacian)
    # Finds sharp regions across the axial slice
    laplacian = ndimage.laplace(inverted)
    energy_map = np.abs(laplacian)
    energy_smooth = ndimage.gaussian_filter(energy_map, sigma=1.5)
    
    # 4. Structural Energy: Meijering Neuriteness Filter
    # sigmas=[1, 2, 3, 4, 5] searches for dendrites of various widths (spines to main shafts)
    neurite_energy = meijering(inverted, sigmas=range(1, 6), black_ridges=False)
    
    # Normalize Meijering output to 0.0 - 1.0
    ne_max = neurite_energy.max()
    if ne_max > 0:
        neurite_energy = neurite_energy / ne_max
        
    # 5. Combine and Threshold
    # Multiplies focus energy with structural neurite energy
    combined_energy = energy_smooth * neurite_energy
    
    thresh_val = threshold_otsu(combined_energy)
    binary_mask = combined_energy > thresh_val
    
    # 6. Morphological Clean-up
    # Merges gaps in paths and eliminates tiny staining artifacts
    cleaned_mask = closing(binary_mask, disk(2))
    cleaned_mask = remove_small_objects(cleaned_mask, min_size=50)
    
    return (cleaned_mask * 255).astype(np.uint8)


# =====================================================================
#  EXECUTION BLOCK: CHANGE THESE PATHS TO MATCH YOUR COMPUTER
# =====================================================================

# 1. Load your input image (Change 'golgi_slice_z10.tif' to your actual file name)
input_path = 'fmtest_sample_data1.png' 
original_image = iio.imread(input_path)

# 2. Run the Meijering segmentation mask pipeline
final_mask = segment_golgi_dendrite_meijering(original_image)

# 3. SAVE the mask image to your folder
output_path = 'golgi_dendrite_mask.png'
iio.imwrite(output_path, final_mask)
print(f"Success! Mask image successfully saved to: {output_path}")

# 4. DISPLAY the results side-by-side on your screen
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Left side: Original Image (Index 0)
axes[0].imshow(original_image, cmap='gray' if original_image.ndim == 2 else None)
axes[0].set_title("Original Golgi-Cox Image")
axes[0].axis('off')

# Right side: Generated Neurite Mask (Index 1)
axes[1].imshow(final_mask, cmap='gray')
axes[1].set_title("Generated Meijering Mask")
axes[1].axis('off')

plt.tight_layout()
plt.show()  # Forces the UI window to pop up and stay open