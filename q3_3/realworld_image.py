from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../python')
from conv_net import convnet_forward
from utils import get_lenet
from init_convnet import init_convnet
from scipy.io import loadmat

def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)

    img = ImageOps.autocontrast(img)

    img = img.resize((28, 28))
    
    img = np.array(img)
    img = img / 255.0

    img = img.reshape(28, 28, 1)
    img = img[np.newaxis, :]

    return img


image_paths = [
    '1.png',
    '2.png',
    '3.png',
    '4.png',
    '5.png',
    '6.png',
    '7.png'
]

# Load the model architecture
layers = get_lenet()
params = init_convnet(layers)

# Load the network
data = loadmat('../results/lenet.mat')
params_raw = data['params']

for params_idx in range(len(params)):
    raw_w = params_raw[0,params_idx][0,0][0]
    raw_b = params_raw[0,params_idx][0,0][1]
    assert params[params_idx]['w'].shape == raw_w.shape, 'weights do not have the same shape'
    assert params[params_idx]['b'].shape == raw_b.shape, 'biases do not have the same shape'
    params[params_idx]['w'] = raw_w
    params[params_idx]['b'] = raw_b

fig, axes = plt.subplots(1, len(image_paths), figsize=(15, 5))

for i, image_path in enumerate(image_paths):
    real_world_image = preprocess_image(image_path)

    # Adjust batch size to match single image input
    layers[0]['batch_size'] = real_world_image.shape[0]

    cptest, P = convnet_forward(params, layers, real_world_image, test=True)
    pred = np.argmax(P, axis=0)

    img_display = Image.open(image_path)

    axes[i].imshow(img_display, cmap='gray')
    axes[i].axis('off')
    axes[i].set_title(f"Pred: {pred[0]}")

plt.show()
