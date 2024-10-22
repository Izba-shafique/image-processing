import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import os
import random

# Define three classes of images
classes = {
    'class1': ['C:/Users/Izba Shafique/Desktop/image processing/sun1.jpg',
               'C:/Users/Izba Shafique/Desktop/image processing/sun2.jpg',
               'C:/Users/Izba Shafique/Desktop/image processing/sun3.jpg',
               'C:/Users/Izba Shafique/Desktop/image processing/sun4.jpg',
               'C:/Users/Izba Shafique/Desktop/image processing/sun5.jpg'],
    'class2': ['C:/Users/Izba Shafique/Desktop/image processing/flower1.jpg',
               'C:/Users/Izba Shafique/Desktop/image processing/flower2.jpg',
               'C:/Users/Izba Shafique/Desktop/image processing/flower3.jpg',
               'C:/Users/Izba Shafique/Desktop/image processing/flower4.jpg',
               'C:/Users/Izba Shafique/Desktop/image processing/flower5.jpg'],
    'class3': ['C:/Users/Izba Shafique/Desktop/image processing/car1.jpg',
               'C:/Users/Izba Shafique/Desktop/image processing/car2.jpg',
               'C:/Users/Izba Shafique/Desktop/image processing/car3.jpg',
               'C:/Users/Izba Shafique/Desktop/image processing/car4.jpg',
               'C:/Users/Izba Shafique/Desktop/image processing/car5.jpg']
}


def get_texture_features(image_path):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Image {image_path} not found.")

    # Calculate GLCM
    glcm = graycomatrix(img, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], symmetric=True, normed=True)

    # Calculate GLCM properties
    contrast = graycoprops(glcm, 'contrast')
    dissimilarity = graycoprops(glcm, 'dissimilarity')
    homogeneity = graycoprops(glcm, 'homogeneity')
    energy = graycoprops(glcm, 'energy')
    correlation = graycoprops(glcm, 'correlation')

    # Flatten and concatenate features
    features = np.concatenate([contrast, dissimilarity, homogeneity, energy, correlation]).flatten()

    return features


def euclidean_distance(feat1, feat2):
    return np.sqrt(np.sum((feat1 - feat2) ** 2))


# Select a reference image and compute its texture
reference_class = random.choice(list(classes.keys()))
reference_image = random.choice(classes[reference_class])
reference_image_path = os.path.join(reference_class, reference_image)
reference_features = get_texture_features(reference_image_path)

print(f"Reference image: {reference_image} from {reference_class}")


def classify_image():
    # Randomly select an image from any class
    random_class = random.choice(list(classes.keys()))
    random_image = random.choice(classes[random_class])
    random_image_path = os.path.join(random_class, random_image)

    # Compute texture features of the random image
    random_features = get_texture_features(random_image_path)

    # Compare textures
    distance = euclidean_distance(reference_features, random_features)

    print(f"Randomly selected image: {random_image}")
    print(f"Actual class: {random_class}")
    print(f"Texture similarity (lower is more similar): {distance}")

    # Determine the class based on texture similarity
    threshold = 10.0  # Define an appropriate threshold
    if distance < threshold:
        print(f"Classified as: {reference_class}")
    else:
        print("Classified as: Different class")

    print("-----")


# Run the classification multiple times
for _ in range(5):
    classify_image()
