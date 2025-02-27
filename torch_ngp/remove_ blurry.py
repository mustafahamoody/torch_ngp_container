import cv2
import os
import argparse


#Set Image Folder Path and Blur Threshold
data_path = arg.data
blur_threshold = 100

def is_blurry(img_path, threshold=blur_threshold):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # Load image as grayscale 
    if image is None:
        return False # Skip non-image files in folder
    
    laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
    return laplacian_var < threshold
    
def remove_blurry(data_path):
    for filename in os.listdir(data_path):
        file_path = os.path.join(data_path, filename)
        if os.path.isfile(file_path) and is_blurry(file_path):
            print(f'Removing Blurry image: {filename}')
            # blurry_images += 1
            os.remove(filepath)
            
if __name__ = __main__:
    remove_blurry(data_path)