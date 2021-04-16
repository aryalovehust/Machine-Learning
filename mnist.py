import numpy as np
import mnist
np.random.seed(42)
# Loading the training and testing data:
X_train, y_train = mnist.train_images(), mnist.train_labels()
X_test, y_test = mnist.test_images(), mnist.test_labels()
num_classes = 10
# classes are the digits from 0 to 9
# We transform the images into column vectors (as inputs for our NN):
X_train, X_test = X_train.reshape(-1, 28*28), X_test.reshape(-1, 28*28)
# We "one-hot" the labels (as targets for our NN), for instance, transform
#label `4` into vector `[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]`:
y_train = np.eye(num_classes)[y_train]