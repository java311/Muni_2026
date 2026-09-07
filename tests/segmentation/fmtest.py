from skimage import io
from skimage.filters import frangi, meijering
import matplotlib.pyplot as plt

# Load image
image = io.imread("fmtest_sample_data1.png")

# Apply filters
frangi_filtered = frangi(image)
meijering_filtered = meijering(image)

# Show results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(image, cmap="gray")
axes[0].set_title("Original")
axes[1].imshow(frangi_filtered, cmap="gray")
axes[1].set_title("Frangi")
axes[2].imshow(meijering_filtered, cmap="gray")
axes[2].set_title("Meijering")
for ax in axes:
    ax.axis("off")
plt.show()