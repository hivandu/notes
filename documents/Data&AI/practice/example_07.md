# Advanced Deep Learning

## Different optimer

```python
import numpy as np
import torch

x = np.random.random(size=(100, 8))
linear = torch.nn.Linear(in_features=8, out_features=1)
sigmoid = torch.nn.Sigmoid()
linear2 = torch.nn.Linear(in_features=1, out_features=1)

model = torch.nn.Sequential(linear, sigmoid, linear2).double()
train_x = torch.from_numpy(x)

print(model(train_x).shape)

yture = torch.from_numpy(np.random.uniform(0, 5, size=(100, 1)))

# print(x)
print(yture.shape)
"""
torch.Size([100, 1])
torch.Size([100, 1])
"""

loss_fn = torch.nn.MSELoss()
optimer = torch.optim.SGD(model.parameters(), lr=1e-5)

for e in range(100):
    for b in range(100 // 1): # stochastic gradient descent
    # for b in range(100 // 10): # mini-batch gradient descent
    # for b in range(100 // 100): # batch gradient descent
        batch_index = np.random.choice(range(len(train_x)), size=20)

        yhat = model(train_x[batch_index])
        loss = loss_fn(yhat, yture[batch_index])
        loss.backward()
        print(loss)
        optimer.step()

"""
tensor(5.0873, dtype=torch.float64, grad_fn=<MseLossBackward>)
tensor(3.4337, dtype=torch.float64, grad_fn=<MseLossBackward>)
show more (open the raw output data in a text editor) ...

tensor(2.1481, dtype=torch.float64, grad_fn=<MseLossBackward>)
"""


```



## Matrix dimension

```python
from torch import nn
import torch
import numpy as np

x = torch.from_numpy(np.random.random(size=(4, 10)))
print(x.shape)
"""
torch.Size([4, 10])
"""

model = nn.Sequential(
    nn.Linear(in_features=10, out_features=5).double(),
    nn.Sigmoid(),
    nn.Linear(in_features=8, out_features=8).double(),
    nn.Sigmoid(),
    nn.Linear(in_features=8, out_features=8).double(),
    nn.Sigmoid(),
    nn.Linear(in_features=8, out_features=8).double(),
    nn.Sigmoid(),
    nn.Linear(in_features=8, out_features=8).double(),
    nn.Sigmoid(),
    nn.Linear(in_features=8, out_features=8).double(),
    nn.Sigmoid(),
    nn.Linear(in_features=8, out_features=8).double(),
    nn.Sigmoid(),
    nn.Linear(in_features=8, out_features=8).double(),
    nn.Sigmoid(),
    nn.Linear(in_features=8, out_features=8).double(),
    nn.Sigmoid(),
    nn.Linear(in_features=8, out_features=8).double(),
    nn.Sigmoid(),
    nn.Linear(in_features=8, out_features=8).double(),
    nn.Softmax()
)

ytrue = torch.randint(8, (4, ))
print(ytrue)
"""
tensor([4, 0, 7, 7])
"""

loss_fn = nn.CrossEntropyLoss()

print(model(x).shape)
print(ytrue.shape)
loss = loss_fn(model(x), ytrue)

print(torch.randint(5, (3, )))

loss.backward()

for p in model.parameters():
    print(p, p.grad)
```


## Advanced deep learning

```python
# Basic computing library
import numpy as np
# Deep learning library
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
# Auxiliary drawing gallery
import matplotlib.pyplot as plt
# Time operation library
import time
# Progress bar control library
from tqdm import tqdm
```



### Project 1: Forward propagation of simple neural network

#### Question 1: Define the initial parameters and activation function

You need to use numpy to implement the forward propagation process of the neural network and calculate the final output result of the output layer. In order to complete the above tasks, we need to make the following assumptions:
1. The value entered is \[3,5\]
1. The two weights of the hidden layer h1 are \[2,4\], \[4,-5\]
1. The two weights of the hidden layer h2 are \[-1,1\], \[2,2\]
1. The weight of the output layer is \[-3,7\]
1. All layers do not use bias
1. All hidden layers need to add tanh activation function

```python
# TODO: Define a numpy array with the input data of the neural network:
input_data = np.array([3, 5])

# TODO: Define a numpy array with the content of the hidden layer and output layer weights of the neural network:
# Tips: The weight dictionary has been built, you only need to fill in the corresponding value according to the hidden layer name
weights = {'h11': np.array([2, 4]),
           'h12': np.array([4, -5]),
           'h21': np.array([-1, 1]),
           'h22': np.array([2, 2]),
           'out': np.array([-3, 7])}

# TODO: Improve the following tanh activation function:
def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
  

```



#### Question 2: Calculate the neural network output layer by layer

In the calculation of the neural network, it is necessary to first multiply the weight of the layer to be calculated with its input data, and then sum, and then through the operation of the activation function, it can be output to the next layer. 

Below we will use the layer as the unit to perform calculations: 

1. The first is the first hidden layer. You need to multiply, sum, and input the data of the input layer and the weight of the hidden layer into the activation function. 

```python
print(input_data * weights['h11'])
a = tanh(input_data * weights['h11']).sum()
b = tanh((input_data * weights['h11']).sum())
print(a,b)
"""
[ 6 20]
1.9999877116507956 1.0
"""

# TODO: multiply, sum, and input the data of the input layer and the weight of the first hidden layer into the activation function.
hidden_11_value = tanh(input_data * weights['h11']).sum()
hidden_12_value = tanh(input_data * weights['h12']).sum()
hidden_1_output = np.array([hidden_11_value, hidden_12_value])
"""
1.9999877116507956
-7.550282621338056e-11
[ 1.99998771e+00 -7.55028262e-11]
"""
```

2. Next is the second hidden layer, the operation of this layer is exactly the same as the previous layer. 

```python
# TODO: multiply, sum, and input the data output by the upper layer and the weight of the second hidden layer into the activation function.
hidden_21_value = tanh(hidden_1_output * weights['h21']).sum()
hidden_22_value = tanh(hidden_1_output * weights['h22']).sum()
hidden_2_output = np.array([hidden_21_value, hidden_22_value])

```

3. Finally, there is the output layer. At this time, there is only one node that needs to be calculated, and there is no need to add an activation function. 

```python
# TODO: multiply and sum the data output by the upper layer and the weight of the output layer
output = (hidden_2_output * weights['out']).sum()
```

4. At this point, you have completed all the calculations. Now let's print out the output of these layers and have a look. 



```python
print(output)
"""
9.887385002294863
"""
```



### Project 2: CIFAR-10 Image Classification

#### Preparation

The data set used in this project can be directly exported from the torchvision library. Here are some basic data operations (data download may take a few minutes, please be patient). 

```python
##Define various transformation operations on the image, including converting the array to tensor, and regularizing the image
#transforms.Compose is mainly used for some common graphics transformations, such as cropping and rotation
#Traverse the list array and perform each transforms operation on the img in turn
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.48216, 0.44653),
                                                     (0.24703, 0.24349, 0.26159))))
#Export the CIFAR10 data set in torchvision. The root is the directory where the data is stored after downloading. The train controls whether it is in the training phase, the download controls whether it needs to be downloaded, and the transform passes in a series of image transformations.
trainset = torchvision.datasets.CIFAR10(root='~/data/course_data/',
                                        train=True,
                                        download=True,
                                        transform=transform)
testset = torchvision.datasets.CIFAR10(root='~/data/course_data/',
                                       train=False,
                                       download=True,
                                       transform=transform)
#Used to divide the training data into multiple groups, this function throws a group of data each time.
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=16,
                                          shuffle=True)
#Used to divide the test data into multiple groups, this function throws a group of data each time.
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=16,
                                         shuffle=False)
"""
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ~/data/course_data/cifar-10-python.tar.gz
170499072it [02:24, 1181561.38it/s]                               
Extracting ~/data/course_data/cifar-10-python.tar.gz to ~/data/course_data/
Files already downloaded and verified
"""
```

After the data download is complete, we can simply check the data label to see if it is correct with the data set in the exercise description. 

```python
trainset.classes
"""
['airplane',
 'automobile',
 'bird',
 'cat',
 'deer',
 'dog',
 'frog',
 'horse',
 'ship',
 'truck']
"""
```

Let's check the data image again.

```python
#Display the pictures visually
#Define drawing function
def imshow(inp, title = None):
    """Imshow for Tensor."""

    # Define the canvas for drawing
    fig = plt.figure(figsize = (30, 30))

    # Convert the dimensions of the picture
    inp = inp.numpy().transpose((1,2,0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Standardize the picture
    inp = std * inp + mean

    # The value of the entire image array is limited to the specified value a_min, and a_max
    inp = np.clip(inp, 0, 1)

    # Visual display of pictures
    plt.imshow(inp,)

# Get a batch of data
inputs, classes = next(iter(trainloader))

# Display in grid format, the function is to combine several images into one image
out = torchvision.utils.make_grid(inputs)

# plt.imshow() can display the picture and also display its format
imshow(out, title = [trainset.classes[x] for x in classes])
```

#### Question 1: Build a simple neural network
After the data is ready, you need to build a simple neural network. 

```python 
# TODO: define a layer 3 fully connected neural network, the input dimension is 32*32*3, the output dimension of the first layer is 1000, the output dimension of the second layer is 500, and the output dimension of the third layer is 10
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 1000) 
        self.fc2 = nn.Linear(1000, 500) 
        self.fc3 = nn.Linear(500, 10) 

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)

# Instantiate the neural network class
net = Net()
```

After the model structure is defined, the loss function and optimizer need to be determined.

```python
# Define loss function-cross entropy
criterion = nn.CrossEntropyLoss()

# Define the optimizer, pass the parameters of the neural network to the optimizer, and define the learning rate
optimizer = optim.Adam(net.parameters(), lr = 3e-4)
```

#### Question 2: Neural Network Training

The main content of the model has been completed, and the training can be carried out below. In the process of model training, the following steps are generally followed:

1. Big for loop-epochs, used to manage a set of data loop training several times
1. Small for loop-step, used to retrieve data from dataloader in batchsize unit
1. Clear the gradient of the optimizer
1. Read in data and label, and perform shape transformation (can be done or not)
1. Run the forward propagation process of the model
1. Generate the final result based on the model output
1. Calculate the loss
1. Calculate the gradient based on the loss
1. Update parameters based on gradient

```python
# TODO: training model
num_epochs = 10
since = time.time()
net.train()

for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1} / {num_epochs}')

    running_loss = 0.0
    running_corrects = 0

    # Take out each batch of data in a loop from the trainloader
    for data in tqdm(trainloader):
        # TODO: Completion code
        inputs, labels = data
        inputs = inputs.view(-1, 32 * 32 * 3)
        optimizer.zero_grad()
        outputs = net(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Calculation of the loss function of a batch of data
        running_loss += loss.item() * inputs.size(0)

        # Calculation of the accuracy of a batch of data
        running_corrects += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / trainloader.dataset.data.shape[0]
    epoch_acc = running_corrects.double() / trainloader.dataset.data.shape[0]

    print('train loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    print('-' * 20)

time_elapsed = time.time()-since
print('Trainning complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed% 60))

"""
Epoch 1 / 10
100%|██████████| 3125/3125 [01:04<00:00, 48.74it/s]
train loss: 1.6377 Acc: 0.4185
--------------------
Epoch 2 / 10
100%|██████████| 3125/3125 [01:04<00:00, 48.15it/s]
train loss: 1.4254 Acc: 0.4962
--------------------
Epoch 3 / 10
100%|██████████| 3125/3125 [01:06<00:00, 47.29it/s]
train loss: 1.3065 Acc: 0.5372
--------------------
Epoch 4 / 10
100%|██████████| 3125/3125 [01:04<00:00, 48.76it/s]
train loss: 1.2026 Acc: 0.5729
--------------------
Epoch 5 / 10
100%|██████████| 3125/3125 [01:02<00:00, 49.98it/s]
train loss: 1.1129 Acc: 0.6033
--------------------
Epoch 6 / 10
100%|██████████| 3125/3125 [01:01<00:00, 51.17it/s]
train loss: 1.0252 Acc: 0.6343
--------------------
Epoch 7 / 10
100%|██████████| 3125/3125 [01:02<00:00, 49.67it/s]
train loss: 0.9373 Acc: 0.6668
--------------------
Epoch 8 / 10
100%|██████████| 3125/3125 [01:02<00:00, 49.63it/s]
train loss: 0.8545 Acc: 0.6936
--------------------
Epoch 9 / 10
100%|██████████| 3125/3125 [01:02<00:00, 50.02it/s]
train loss: 0.7770 Acc: 0.7242
--------------------
Epoch 10 / 10
100%|██████████| 3125/3125 [01:02<00:00, 50.16it/s]train loss: 0.7020 Acc: 0.7492
--------------------
Trainning complete in 10m 33s
"""
```

#### Question 3: Model evaluation

After completing the model training, the model needs to be evaluated to verify the accuracy of the model on the test set.

Tips: In the model training log, the accuracy acc is also printed, but this is the accuracy of the model on the training set, not the accuracy on the test set. You can observe the accuracy of the training set and the accuracy of the test set to see if there is any difference.

```python
# TODO: Complete model evaluation
correct, total = 0, 0
net.eval()

for data in tqdm(testloader):
    inputs, labels = data
    inputs = inputs.view(-1, 32 * 32 * 3)
    outputs = net(inputs)
    _, predicted = torch.max(outputs, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print('The testing set accuracy of the network is: %d %%'% (100 * correct / total))
"""
100%|██████████| 625/625 [00:03<00:00, 157.71it/s]The testing set accuracy of the network is: 53 %
"""
```

