# Business Intelligence(BI)

## Use LeNet model to recognize Mnist handwritten digits


```python
import tensorflow as tf
#print(tf.__version__)
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Data loading
#(train_x, train_y), (test_x, test_y) = mnist.load_data() #Download the data set from the Internet
data = np.load('~/data/course_data/mnist.npz') #Read data set from local
#print(data.files)
train_x, train_y, test_x, test_y = data['x_train'], data['y_train'], data['x_test'], data['y_test']

warnings.filterwarnings('ignore')
# Input data is mnist data set
train_x = train_x.reshape(train_x.shape[0], 28, 28, 1)
test_x = test_x.reshape(test_x.shape[0], 28, 28, 1)
train_x = train_x / 255
test_x = test_x / 255
train_y = tf.keras.utils.to_categorical(train_y, 10)
test_y = tf.keras.utils.to_categorical(test_y, 10)
# Create sequential model
model = Sequential()
# The first layer of convolutional layer: 6 convolution kernels, the size is 5*5, relu activation function
model.add(Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)))
# The second pooling layer: maximum pooling
model.add(MaxPooling2D(pool_size=(2, 2)))
# The third layer of convolutional layer: 16 convolution kernels, size 5*5, relu activation function
model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
# The second pooling layer: maximum pooling
model.add(MaxPooling2D(pool_size=(2, 2)))
# Flatten the parameters, which is called a convolutional layer in LeNet5. In fact, this layer is a one-dimensional vector, the same as the fully connected layer
model.add(Flatten())
model.add(Dense(120, activation='relu'))
# Fully connected layer, the number of output nodes is 84
model.add(Dense(84, activation='relu'))
# The output layer uses the softmax activation function to calculate the classification probability
model.add(Dense(10, activation='softmax'))
# Set the loss function and optimizer configuration
model.compile(loss=tf.keras.metrics.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
# Pass in training data for training
model.fit(train_x, train_y, batch_size=128, epochs=2, verbose=1, validation_data=(test_x, test_y))
# Evaluate the results
score = model.evaluate(test_x, test_y)
print('error:%0.4lf' %score[0])
print('Accuracy:', score[1])

"""
Train on 60000 samples, validate on 10000 samples
Epoch 1/2
60000/60000 [==============================] - 39s 643us/sample - loss: 0.3172 - acc: 0.9096 - val_loss: 0.1105 - val_acc: 0.9626
Epoch 2/2
60000/60000 [==============================] - 39s 652us/sample - loss: 0.0892 - acc: 0.9725 - val_loss: 0.0664 - val_acc: 0.9790
10000/10000 [==============================] - 4s 358us/sample - loss: 0.0664 - acc: 0.9790
error:0.0664
Accuracy: 0.979
"""
```

##  Use LR to classify MNIST handwritten digits

```python
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Download Data
digits = load_digits()
data = digits.data
# Data Exploration
print(data.shape)
# View the first image
print(digits.images[0])
# The meaning of the numbers represented by the first image
print(digits.target[0])
# Display the first image
plt.gray()
plt.title('Handwritten Digits')
plt.imshow(digits.images[0])
plt.show()
"""
(1797, 64)
[[ 0.  0.  5. 13.  9.  1.  0.  0.]
 [ 0.  0. 13. 15. 10. 15.  5.  0.]
 [ 0.  3. 15.  2.  0. 11.  8.  0.]
 [ 0.  4. 12.  0.  0.  8.  8.  0.]
 [ 0.  5.  8.  0.  0.  9.  8.  0.]
 [ 0.  4. 11.  0.  1. 12.  7.  0.]
 [ 0.  2. 14.  5. 10. 12.  0.  0.]
 [ 0.  0.  6. 13. 10.  0.  0.  0.]]
0
"""

```

![image-20210901154812986](http://qiniu.hivan.me/MAMTimage-20210901154812986.png?img)



```python
# Split the data, use 25% of the data as the test set, and the rest as the training set
train_x, test_x, train_y, test_y = train_test_split(data, digits.target, test_size=0.25, random_state=33)

# Adopt Z-Score standardization
ss = preprocessing.StandardScaler()
train_ss_x = ss.fit_transform(train_x)
test_ss_x = ss.transform(test_x)

# Create LR classifier
lr = LogisticRegression()
lr.fit(train_ss_x, train_y)
predict_y=lr.predict(test_ss_x)
print('LR accuracy rate: %0.4lf'% accuracy_score(predict_y, test_y))
"""
LR accuracy rate: 0.9644
"""
```

