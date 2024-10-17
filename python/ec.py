# Notice: change the value in line 58 to get the correct output accordingly
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.io import loadmat

sys.path.insert(0, '../python')
from conv_net import convnet_forward
from init_convnet import init_convnet
from utils import get_lenet

layers = get_lenet(batch_size=1)
params = init_convnet(layers)

data = loadmat('../results/lenet.mat')
params_raw = data['params']
for params_idx in range(len(params)):
    raw_w = params_raw[0, params_idx][0, 0][0]
    raw_b = params_raw[0, params_idx][0, 0][1]
    params[params_idx]['w'] = raw_w
    params[params_idx]['b'] = raw_b

image_paths = ["../images/image1.JPG", 
               "../images/image2.JPG", 
               "../images/image3.png",
               "../images/image4.jpg"]

for idx, image_path in enumerate(image_paths):
    print(f"Processing Image {idx+1}")

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    thresholded_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY_INV, 11, 2)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresholded_image)

    processed_digits = []

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area > 58:  # Change this value to include or exclude digits or noises in different shape
            # use 60 for image1 and 58 for image4
            digit_image = thresholded_image[y:y+h, x:x+w]
            
            padding_height = int(h / 5)
            padding_width = int(w / 5)
            digit_image_padded = cv2.copyMakeBorder(digit_image, 
                                                    padding_height, padding_height,
                                                    padding_width, padding_width,
                                                    cv2.BORDER_CONSTANT, value=[0])
            digit_image_resized = cv2.resize(digit_image_padded, (28, 28), interpolation=cv2.INTER_AREA)

            digit_image_resized = digit_image_resized / 255.0
            digit_input = digit_image_resized.reshape(28 * 28, 1)

            _, predictions = convnet_forward(params, layers, digit_input, test=True)
            predicted_digit = np.argmax(predictions, axis=0)

            processed_digits.append((digit_image_resized, predicted_digit))
    
    num_cols = 10
    num_rows = (len(processed_digits) + num_cols - 1) // num_cols

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 2))
    axs = axs.flatten()
    
    for j, (digit_img, predicted_digit) in enumerate(processed_digits):
        axs[j].imshow(digit_img, cmap='gray')
        axs[j].set_title(f"Pred: {predicted_digit}")
        axs[j].axis('off')

    for j in range(len(processed_digits), num_rows * num_cols):
        axs[j].axis('off')

    plt.suptitle(f"Results for Image {idx+1}")
    plt.show()
