import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def contrast_stretching(image):

    fifth = np.percentile(image, 5)
    ninth = np.percentile(image, 95)
    stretched = image.copy()

    for i in range(stretched.shape[0]):
        for j in range(stretched.shape[1]):
            if stretched[i][j] <= fifth:
                stretched[i][j] = 0
            elif stretched[i][j] >= ninth:
                stretched[i][j] = 255
            else:
                stretched[i][j] = 255 * (stretched[i][j] - fifth) / (ninth - fifth)
    return np.uint8(stretched)

def dice_coefficient(segmented, original):
    intersection = np.sum((segmented > 0) & (original > 0))
    total_pixels = np.sum(segmented > 0) + np.sum(original > 0)
    return 2 * intersection / total_pixels 

def count_true_false_pixels(segmented, original):
    true_pixels = np.sum(segmented == original)  
    false_pixels = np.sum(segmented != original) 
    return true_pixels, false_pixels

def segment_cells(image):

    enhanced_gray = contrast_stretching(image)
    mask_white = cv2.inRange(enhanced_gray, 0, 60) 
    mask_gray = cv2.inRange(enhanced_gray, 60, 200) 
    
    vset = set(range(0, 200)) 
    largest_component = get_largest_component(enhanced_gray, vset)
    cv2.imshow("Largest Component", largest_component)
    cv2.waitKey(0)

    final_image = np.zeros_like(image)  
    final_image[largest_component > 0] = 127  
    cv2.imshow("Final Image", final_image)
    cv2.waitKey(0)
    final_image[mask_white > 0] = 255  
    cv2.imshow("Final Image", final_image)
    cv2.waitKey(0)
    return final_image

def process_images(image_path, mask_path, output_folder,image_index):

    os.makedirs(output_folder, exist_ok=True)
    image_files = sorted(os.listdir(image_path))
    mask_files = sorted(os.listdir(mask_path))
    
    total_dice = 0
    num_images = 0
    
    for i, (img_name, mask_name) in enumerate(zip(image_files, mask_files)):
        img_file = os.path.join(image_path, img_name)
        mask_file = os.path.join(mask_path, mask_name)
         
        image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        
        final_image = segment_cells(image)
        dice_score = dice_coefficient(final_image, mask)
        true_pixels, false_pixels = count_true_false_pixels(final_image, mask)
        
        total_dice += dice_score
        num_images += 1
        
        output_file = os.path.join(output_folder, f"segmented_{img_name}")
        cv2.imwrite(output_file, final_image)
        
        print(f"Processed {img_name} - Dice Coefficient: {dice_score:.4f}, True Pixels: {true_pixels}, False Pixels: {false_pixels}")
        
        if image_index is not None and i == image_index:
            plot_segmentation(image, mask, final_image)
            break  
    
    average_dice = total_dice / num_images 
    print(f"Average Dice Coefficient: {average_dice:.4f}")

def plot_segmentation(image, mask, segmented_image):

    plt.figure(figsize=(7,6))
    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original Grayscale Image")
    plt.axis("off")
    plt.subplot(2, 2, 2)
    plt.imshow(contrast_stretching(image), cmap='gray')
    plt.title("Enhanced Contrast Image")
    plt.axis("off")
    plt.subplot(2, 2, 3)
    plt.imshow(mask, cmap='gray')
    plt.title("Mask")
    plt.axis("off")
    plt.subplot(2, 2, 4)
    plt.imshow(segmented_image, cmap='gray', vmin=0, vmax=255)
    plt.title("Segmented Output")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

image_path = r"C:\Users\User\Desktop\dataset_DIP_assignment\train\images"
mask_path = r"C:\Users\User\Desktop\dataset_DIP_assignment\train\masks"
output_folder = r"C:\Users\User\Desktop\dataset_DIP_assignment\output"
image_index = 0

process_images(image_path, mask_path, output_folder,image_index)
