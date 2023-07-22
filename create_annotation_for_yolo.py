import cv2
import numpy as np
import os
from PIL import Image
import glob
import argparse
def slide_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def extract_bounding_boxes(mask):
    # find contours in the mask and initialize the bounding box
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # normalize the bounding box coordinates to be between 0 and 1
        xmax = min(1, (x+w)/mask.shape[1])
        xmin = max(0, x/mask.shape[1])
        ymax = min(1, (y+h)/mask.shape[0])
        ymin = max(0, y/mask.shape[0])
        
        bounding_boxes.append((xmin, ymin, xmax, ymax))
    
    return bounding_boxes

def save_bounding_boxes_to_txt(bounding_boxes, filename):
    with open(filename, 'w') as f:
        for bb in bounding_boxes:
            f.write('0 {} {} {} {}\n'.format(*bb))

if __name__  == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_imgs_path')     
    parser.add_argument('--input_masks_path')   
    parser.add_argument('--output_path ')     
    args = parser.parse_args()

    # Set the window size and the step size
    window_size = (1024, 1024)
    step_size = 1024


    # Set the path to your image and mask
    image_post_fix =  "*"  #or '.png'
    img_paths = glob.glob(os.path.join(args.input_imgs_path,image_post_fix))
    mask_paths = glob.glob(os.path.join(args.input_masks_path,image_post_fix))
    
    img_paths.sort()
    mask_paths.sort()
    for (img_path,mask_path) in zip(img_paths,mask_paths):
        # Load the image and mask
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # Slide the window across the image
        for (x, y, window) in slide_window(image, step_size=step_size, windowSize=window_size):
            if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
                continue

            # Extract the corresponding mask
            mask_window = mask[y:y + window_size[1], x:x + window_size[0]]

            # Find the bounding boxes
            bounding_boxes = extract_bounding_boxes(mask_window)

            # Save the bounding boxes to a text file
            filename = "{}_{}_{}.txt".format(x, y, os.path.basename(img_path).split('.')[0])
            save_bounding_boxes_to_txt(bounding_boxes, filename)
