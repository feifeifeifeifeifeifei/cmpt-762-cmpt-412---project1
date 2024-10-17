import numpy as np
from utils import get_lenet
from load_mnist import load_mnist
from scipy.io import loadmat
from conv_net import convnet_forward
from init_convnet import init_convnet
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

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

# Load data
fullset = False
xtrain, ytrain, xvalidate, yvalidate, xtest, ytest = load_mnist(fullset)

ytest = np.squeeze(ytest)
# Testing the network
#### Modify the code to get the confusion matrix ####
all_preds = []
for i in range(0, xtest.shape[1], 100):
    cptest, P = convnet_forward(params, layers, xtest[:, i:i+100], test=True)
    preds = np.argmax(P, axis=0)
    all_preds.extend(preds)

all_preds = np.array(all_preds)
all_labels = ytest

cm = confusion_matrix(all_labels, all_preds)

print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for MNIST Test Set')
plt.show()

cm_no_diag = cm.copy()
np.fill_diagonal(cm_no_diag, 0)

max_indices = np.unravel_index(np.argsort(cm_no_diag, axis=None)[-2:], cm_no_diag.shape)
confused_pairs = list(zip(max_indices[0], max_indices[1]))

print("The two most confused class pairs:")
for i, j in confused_pairs:
    print(f"Class {i} was misclassified as {j} {cm[i, j]} times")
# Here you can analyze the pairs manually based on the images in MNIST dataset.
# hint: 
#     you can use confusion_matrix from sklearn.metrics (pip install -U scikit-learn)
#     to compute the confusion matrix. Or you can write your own code :)

