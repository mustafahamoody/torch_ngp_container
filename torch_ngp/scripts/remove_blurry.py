import cv2
import os
import argparse

parser = argparse.ArgumentParser(description="Remove blurry images from a folder")
parser.add_argument('--data', type=str, required=True, help='Path to the images folder')
parser.add_argument('--blur_threshold', type=int, default=100, help='Blur threshold')
args = parser.parse_args()

#Set Image Folder Path and Blur Threshold
data_path = args.data
blur_threshold = args.blur_threshold

images_counter = 0
blurry_images_counter = 0

def is_blurry(img_path, threshold=blur_threshold, images_counter=images_counter):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # Load image as grayscale 
    if img is None:
        return False # Skip non-image files in folder
    
    images_counter += 1

    laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var() # Calculate Laplacian variance to determine how blurry the image is
    return laplacian_var < threshold
    
def remove_blurry(data_path, images_counter=images_counter, blurry_images_counter=blurry_images_counter):
    for filename in os.listdir(data_path):
        img_path = os.path.join(data_path, filename)
        if os.path.isfile(img_path) and is_blurry(img_path):
            print(f'Removing Blurry image: {filename}')
            blurry_images_counter += 1
            os.remove(img_path)

    print(f'Removed {blurry_images_counter} (out of {images_counter} images)')
    print(f'{blurry_images_counter/images_counter * 100}% of images were detected as blurry')
    print('')
            
if __name__ == '__main__':
    remove_blurry(data_path)