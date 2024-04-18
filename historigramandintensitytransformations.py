import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests
from cv2 import calcHist

# download images 
# URLs of the images to download
urls = [
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/lenna.png",
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/baboon.png",
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/goldhill.bmp",
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/cameraman.jpeg",
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/zelda.png",
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-CV0101EN-SkillsNetwork/images%20/images_part_1/mammogram.png"
]

# Download images
for url in urls:
    response = requests.get(url)
    if response.status_code == 200:
        filename = url.split("/")[-1]
        with open(filename, 'wb') as f:
            f.write(response.content)
    else:
        print(f"Failed to download {url}")
##########################################################################
# Histogram and Intensity Transformations
# Load the image
import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_image(image_1, image_2, title_1="Original", title_2="New Image"):
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(image_1, cmap="gray")
    plt.title(title_1)
    plt.subplot(1, 2, 2)
    plt.imshow(image_2, cmap="gray")
    plt.title(title_2)
    plt.show()

def plot_hist(old_image, new_image, title_old="Original", title_new="New Image"):
    intensity_values = np.array(list(range(256)))
    plt.subplot(1, 2, 1)
    plt.bar(intensity_values, cv2.calcHist([old_image], [0], None, [256], [0, 256])[:, 0], width=5)
    plt.title(title_old)
    plt.xlabel('intensity')
    plt.subplot(1, 2, 2)
    plt.bar(intensity_values, cv2.calcHist([new_image], [0], None, [256], [0, 256])[:, 0], width=5)
    plt.title(title_new)
    plt.xlabel('intensity')
    plt.show()


# HISTOGRAM AND INTENSITY TRANSFORMATIONS
# Load the images
#apply random image generator from urls for the images
 

#toy image
image = cv2.imread("zelda.png", cv2.IMREAD_GRAYSCALE)
new_image = cv2.bitwise_not(image)
# Apply binary thresholding
ret, new_image = cv2.threshold(image, 86, 255, cv2.THRESH_BINARY)
plot_image(image, new_image, "Original", "Image After Binary Thresholding")
plot_hist(image, new_image, "Original", "Image After Binary Thresholding")

# Apply truncation thresholding
ret, new_image = cv2.threshold(image, 86, 255, cv2.THRESH_TRUNC)
plot_image(image, new_image, "Original", "Image After Truncation Thresholding")
plot_hist(image, new_image, "Original", "Image After Truncation Thresholding")

# Apply Otsu's thresholding
ret, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
plot_image(image, otsu, "Original", "Otsu")
plot_hist(image, otsu, "Original", "Otsu's method")

print(ret)
print('~~~~~~~~~~~~end of line~~~~~~~~~~~~~~~')