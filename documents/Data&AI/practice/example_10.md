# Natural Language Processing NLP





## Resnet Visualize

```python
import torchvision
import torch.nn.functional as F
from torchvision.transforms import transforms
from torch import nn
import torch
import matplotlib.pyplot as plt
from icecream import ic
from PIL import Image
import numpy as np

def visualize_model(model, input_, output):
    width = 8
    fig, ax = plt.subplots(output[0].shape[0] // width, width, figsize=(20, 20))

    for i in range(output[0].shape[0]):
        ix = np.unravel_index(i, ax.shape)
        plt.sca(ax[ix])
        ax[ix].title.set_text('filter-{}'.format(i))
        plt.imshow(output[0][i].detach())

    plt.show()

preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

resnet = torchvision.models.resnet18(pretrained=True) # transfer step 1: load pretrained model

conv_model = [m for _, m in resnet.named_modules() if isinstance(m, torch.nn.Conv2d)]
"""
Downloading: "https://download.pytorch.org/models/resnet18-f37072fd.pth" to /Users/lilithgames/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
100%|██████████| 44.7M/44.7M [00:34<00:00, 1.36MB/s]
"""

for m in conv_model:
    m.register_forward_hook(visualize_model)

myself = preprocess(Image.open('~/data/course_data/doo.jpeg'))

with torch.no_grad():
    resnet(myself.unsqueeze(0)) # un-squeeze for convert myself to [ [myself] ]
```

> Only some pictures are posted here

![image-20210901151356189](http://qiniu.hivan.me/MAMTimage-20210901151356189.png?img)



![image-20210901151412761](http://qiniu.hivan.me/MAMTimage-20210901151412761.png?img)



![image-20210901151432743](http://qiniu.hivan.me/MAMTimage-20210901151432743.png?img)



## Transfer Example



```python
import torchvision
import torch.nn.functional as F
from torchvision.transforms import transforms
from torch import nn
import torch
import matplotlib.pyplot as plt
from icecream import ic

preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

cifar_10 = torchvision.datasets.CIFAR10('~/data/course_data/', download=False, transform=preprocess)

train_loader = torch.utils.data.DataLoader(cifar_10, batch_size=128, shuffle=True)

resnet = torchvision.models.resnet18(pretrained=True) # transfer step 1: load pretrained model

for param in resnet.parameters():
    param.requires_grad = False  # frozen weights
    
feature_num = resnet.fc.in_features
resnet.fc = nn.Linear(feature_num, 10)  # rewrite fc classifier

ic(resnet(cifar_10[0][0].unsqueeze(0)))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(resnet.parameters(), lr=1e-3, momentum=0.9)

epochs = 2

losses = []
"""
return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
ic| resnet(cifar_10[0][0].unsqueeze(0)): tensor([[-0.0763, -0.4537,  0.8168,  0.2136, -0.0465,  0.4844, -0.4026,  0.8763,
                                                  -0.7048, -0.7375]], grad_fn=<AddmmBackward>)
"""

for epoch in range(epochs):
    epoch_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        ic(epoch, i)
        output = resnet(images)
        loss = criterion(output, labels)
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

        if i > 0:
            print('Epoch: {} batch:{}, loss ==> {}'.format(epoch, i, epoch_loss / i))

    losses.append(epoch_loss / i)
"""
ic| epoch: 0, i: 0
ic| epoch: 0, i: 1
ic| epoch: 0, i: 2
Epoch: 0 batch:1, loss ==> 5.118020296096802
ic| epoch: 0, i: 3
Epoch: 0 batch:2, loss ==> 3.8235710859298706
ic| epoch: 0, i: 4
...
ic| epoch: 0, i: 203
Epoch: 0 batch:202, loss ==> 1.4433288293899875
...
"""

plt.plot(losses)
plt.show()
"""
Because the last time is too long to run, the losses are not assigned
"""

```



## Resnet Transfer Learning

```python
import torchvision
import torch.nn.functional as F

cifar_10 = torchvision.datasets.CIFAR10('~/data/course_data', download=False, transform=preprocess)
train_loader = torch.utils.data.DataLoader(cifar_10,
                                          batch_size=512,
                                          shuffle=True)

plt.imshow(cifar_10[10][0].permute(1, 2, 0))

for param in res_net.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = res_net.fc.in_features

res_net.fc = nn.Linear(num_ftrs, 10) # only update this part parameters 

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(res_net.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochslosses = []

epochs = 10

for epoch in range(epochs):
    loss_train = 0
    for i, (imgs, labels) in enumerate(train_loader):        
        print(i)
        outputs = res_net(imgs)
        
        loss = criterion(outputs, labels)
        
        optimizer_conv.zero_grad()
        
        loss.backward()
        
        optimizer_conv.step()
        
        loss_train += loss.item()
        
        if i > 0 and i % 10 == 0:
            print('Epoch: {}, batch: {}'.format(epoch, i))
            print('-- loss: {}'.format(loss_train / i))
            
    losses.append(loss_train / len(train_loader))
```



## Show Resnet

```python
import cv2
import numpy as np

import torch
from torchvision import transforms
from torchvision.models import resnet18
from torchsummary import summary
import matplotlib.pyplot as plt

def show_one_model(model, input_, output):
    width = 8

    fig, ax = plt.subplots(output[0].shape[0] // width, width, figsize=(20, 20))

    for i in range(output[0].shape[0]):
        ix = np.unravel_index(i, ax.shape)
        plt.sca(ax[ix])
        ax[ix].title.set_text('Filter-{}'.format(i))
        plt.imshow(output[0][i].detach())
        # plt.pause(0.05)

    input('this is conv: {}, received a {} tensor,  press any key to continue: '.format(model, input_[0].shape))

    plt.show()
    
def main(img):
    """
    Forward propagation, print feature maps during the transfer process
    """

    # Define device, transforms
    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    ])
    # Process pictures, define models
    img = transform(img).unsqueeze(0)
    model = resnet18(pretrained=True)

    # Print model summary, which can be used for convolutional layer comparison
    summary(model, (3, 224, 224))

    for p in model.parameters():
        print(p)

    conv_models = [m for _, m in model.named_modules() if isinstance(m, torch.nn.Conv2d)]

    for conv in conv_models:
        conv.register_forward_hook(show_one_model)

    with torch.no_grad():
        model(img)

    # conv_models = [m for _, m in model.named_modules() if isinstance(m, torch.nn.Conv2d)]
    #
    # first_conv = conv_models[0]
    #
    # show_one_model(first_conv, img, output=first_conv(img))
    
if __name__ == '__main__':
    img = cv2.imread('~/data/course_data/doo.png')
    main(img)
    
"""
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 112, 112]           9,408
       BatchNorm2d-2         [-1, 64, 112, 112]             128
show more (open the raw output data in a text editor) ...

        -2.5093e-02,  6.7847e-03, -1.7868e-02, -7.8250e-04, -6.3448e-03],
       requires_grad=True)
"""
```



![image-20210901152800418](http://qiniu.hivan.me/MAMTimage-20210901152800418.png?img)



![image-20210901152815850](http://qiniu.hivan.me/MAMTimage-20210901152815850.png?img)





![image-20210901152835421](http://qiniu.hivan.me/MAMTimage-20210901152835421.png?img)





