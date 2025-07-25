import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Updated known color to stain mapping (add more if needed)
stain_color_map = {
    'light pink': 'Eosin (cytoplasm)',
    'red': 'Giemsa (cytoplasm)',
    'rosy brown': 'Oxidized hemoglobin',
    'gray': 'Wright stain (mature RBC)',
    'purple': 'Giemsa (nucleus)',
    'blue': 'Methylene blue / Wright-Giemsa (WBC nucleus)',
    'light blue': 'Toluidine blue / hematoxylin',
    'dark blue': 'Hematoxylin (nucleus)',
    'green': 'Malachite green (bacteria)',
    'yellow': 'Sudan stains / lipid',
    'orange': 'Papanicolaou (keratin)',
    'black': 'Melanin or carbon pigment',
    'brown': 'Iron stain / oxidized hemoglobin',
    'gray pink': 'Fixative artifact',
}

# Function to match RGB to closest stain color
def match_color_to_stain(rgb):
    min_dist = float('inf')
    closest_color = "Unknown"
    stain = "Unknown stain"
    for color_name, stain_name in stain_color_map.items():
        # Convert color name to RGB using matplotlib
        try:
            color_rgb = np.array(plt.colors.to_rgb(color_name)) * 255
        except:
            continue
        dist = np.linalg.norm(rgb - color_rgb)
        if dist < min_dist:
            min_dist = dist
            closest_color = color_name
            stain = stain_name
    return closest_color, stain

# Load image
image = cv2.imread(r"D:\dataset_initial\Positives\3.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Crop center circle region manually (customize size for microscope view)
h, w, _ = image.shape
cx, cy = w // 2, h // 2
r = min(h, w) // 3  # radius
mask = np.zeros((h, w), dtype=np.uint8)
cv2.circle(mask, (cx, cy), r, 255, -1)
masked_img = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
pixels = masked_img[mask == 255]

# Use KMeans to detect dominant colors
k = 5
kmeans = KMeans(n_clusters=k, random_state=0).fit(pixels)
unique, counts = np.unique(kmeans.labels_, return_counts=True)
percentages = counts / counts.sum()

# Prepare output
dominant_info = []
for i, center in enumerate(kmeans.cluster_centers_):
    rgb = tuple(map(int, center))
    percent = round(percentages[i] * 100, 2)
    color_name, stain = match_color_to_stain(np.array(rgb))
    dominant_info.append((rgb, percent, color_name.upper(), stain))

# Sort by percentage
dominant_info.sort(key=lambda x: -x[1])

# Print Results
print("\nDetected Dominant Colors (inside microscope circle):")
for i, (rgb, percent, cname, stain) in enumerate(dominant_info):
    print(f"\nColor {i+1}: RGB {rgb} | Confidence: {percent}%")
    print(f"→ Matched Color: {cname}")
    print(f"→ Likely Staining Used: {stain}")

# Most probable stain
top = dominant_info[0]
print("\n✅ Most Probable Stain Color Detected:", top[2], f"({top[1]}%)")

# Plot color bar
fig, ax = plt.subplots(figsize=(8, 1))
bar = np.zeros((50, 300, 3), dtype=np.uint8)
start = 0
for rgb, pct, _, _ in dominant_info:
    width = int(pct * 3)
    end = start + width
    bar[:, start:end] = rgb
    start = end
ax.imshow(bar)
ax.axis('off')
plt.title("Dominant Colors")
plt.tight_layout()
plt.show()
