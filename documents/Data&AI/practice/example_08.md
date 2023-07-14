# RNN
## Simple RNN

### Define function

Import the required libraries

```python
import io
import os
import unicodedata
import string
import glob

import torch
import random
```

```python
# alphabet small + capital letters + ".,;'"
ALL_LETTERS = string.ascii_letters + ".,;'"
N_LETTERS = len(ALL_LETTERS)
```

Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427

```python
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in ALL_LETTERS
    )
```

```python
def load_data():
    # Build the category_lines dictionary, a list of names per language
    category_lines = {}
    all_categories = []

    def find_files(path):
        return glob.glob(path)

    # Read a file and split into lines
    def read_lines(filename):
        lines = io.open(filename, encoding = 'utf-8').read().strip().split('\n')
        return [unicode_to_ascii(line) for line in lines]

    for filename in find_files('~/data/course_data/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)

        lines = read_lines(filename)
        category_lines[category] = lines

    return category_lines, all_categories
```

To represent a single letter, we use a “one-hot vector” of  size <1 x n_letters>. A one-hot vector is filled with 0s except for a 1 at index of the current letter, e.g. "b" = <0 1 0 0 0 ...>.

To make a word we join a bunch of those into a 2D matrix <line_length x 1 x n_letters>.

That extra 1 dimension is because PyTorch assumes everything is in batches - we’re just using a batch size of 1 here.

```python
# Find letter index from all_letters, e.g. "a" = 0
def letter_to_index(letter):
    return ALL_LETTERS.find(letter)
  
# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letter_to_tensor(letter):
    tensor = torch.zeros(1, N_LETTERS)
    tensor[0][letter_to_index(letter)] = 1
    return tensor
  
# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, N_LETTERS)
    for i, letter in enumerate(line):
        tensor[i][0][letter_to_index(letter)] = 1
    return tensor
  
def random_training_example(category_lines, all_categories):
    def random_choice(a):
        random_idx = random.randint(0, len(a) - 1)
        return a[random_idx]

    category = random_choice(all_categories)
    line = random_choice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype = torch.long)
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor
  
if __name__ == '__main__':
    print(ALL_LETTERS)
    print(unicode_to_ascii('Ślusàrski'))

    category_lines, all_categories = load_data()
    print(category_lines['Italian'][:5])

    print(letter_to_tensor('J'))  # [1, 57]
    print(line_to_tensor('Jones').size())  # [5, 1, 57]
    
"""
abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,;'
Slusarski
['Abandonato', 'Abatangelo', 'Abatantuono', 'Abate', 'Abategiovanni']
tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0.]])
torch.Size([5, 1, 56])
"""
```



### Second Example

```python
# Import the required libraries
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class RNN(nn.Module):
    # implement RNN from scratch rather than using nn.RNN
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim = 1)

    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor), 1)

        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

category_lines, all_categories = load_data()
n_categories = len(all_categories)

n_hidden = 128
rnn = RNN(N_LETTERS, n_hidden, n_categories)

# one step
input_tensor = letter_to_tensor('A')
hidden_tensor = rnn.init_hidden()

output, next_hidden = rnn(input_tensor, hidden_tensor)
print(output.size())
print(next_hidden.size())
"""
torch.Size([1, 18])
torch.Size([1, 128])
"""

# whole sequence/name
input_tensor = line_to_tensor('Albert')
hidden_tensor = rnn.init_hidden()

output, next_hidden = rnn(input_tensor[0], hidden_tensor)
print(output.size())
print(next_hidden.size())
"""
torch.Size([1, 18])
torch.Size([1, 128])
"""

def category_from_output(output):
    category_idx = torch.argmax(output).item()
    return all_categories[category_idx]
  
print(category_from_output(output))
"""
German
"""

criterion = nn.NLLLoss()
learning_rate = 0.005
optimizer = torch.optim.SGD(rnn.parameters(), lr = learning_rate)

def train(line_to_tensor, category_tensor):
    hidden = rnn.init_hidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_to_tensor[i], hidden)
    
    loss = criterion(output, category_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return output, loss.item()
  
current_loss = 0
all_losses = []
plot_steps, print_steps = 1000, 5000
n_iters = 100000

for i in range(n_iters):
    category, line, category_tensor, line_tensor = random_training_example(category_lines, all_categories)

    output, loss = train(line_tensor, category_tensor)
    current_loss += loss

    if (i + 1) % plot_steps == 0:
        all_losses.append(current_loss / plot_steps)
        current_loss = 0

    if (i + 1) % print_steps == 0:
        guess = category_from_output(output)
        corrent = 'CORRECT' if guess == category else f'WRONG ({category})'
        print(f'{i+1} {(i+1) / n_iters *100} {loss:.4f} {line} / {guess} {corrent}')
        
"""
5000 5.0 2.5063 Bureau / Scottish WRONG (French)
10000 10.0 1.4726 Bitar / Arabic CORRECT
15000 15.0 1.9405 Bazilevitch / Russian CORRECT
20000 20.0 1.5565 Dupont / French CORRECT
25000 25.0 0.1202 Majewski / Polish CORRECT
30000 30.0 1.1579 Kucharova / Czech CORRECT
35000 35.0 1.0075 Sheng / Chinese CORRECT
40000 40.0 0.8343 Masih / Arabic CORRECT
45000 45.0 0.5371 Fan / Chinese CORRECT
50000 50.0 0.3260 Vinh / Vietnamese CORRECT
55000 55.00000000000001 2.5464 Pahlke / Polish WRONG (German)
60000 60.0 1.5921 Clark / Scottish CORRECT
65000 65.0 4.3648 Paulis / Greek WRONG (Dutch)
70000 70.0 1.3289 Thian / Vietnamese WRONG (Chinese)
75000 75.0 2.2715 Kelly / English WRONG (Irish)
80000 80.0 1.0069 Siu / Korean WRONG (Chinese)
85000 85.0 0.8168 Kan / Chinese CORRECT
90000 90.0 0.2283 Dinh / Vietnamese CORRECT
95000 95.0 2.0048 Abbascia / Japanese WRONG (Italian)
100000 100.0 0.6310 O'Shea / Irish CORRECT
"""

plt.figure()
plt.plot(all_losses)
plt.show()
```

![image-20210901183345054](http://qiniu.hivan.me/MAMTimage-20210901183345054.png?img)



```python
def predict(input_line):
    print(f'\n > {input_line}')
    with torch.no_grad():
        line_tensor = line_to_tensor(input_line)

        hidden = rnn.init_hidden()

        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)

        guess = category_from_output(output)
        print(guess)
        
while True:
    sentence = input('Input: ')
    if sentence == 'quit':
        break
    predict(sentence)
"""
 > Chinese
Irish

 > English
English

 > Japanese
French

 > French
German
"""
```



## LSTM Modeling trigonometric functions

### Use LSTM to fit sine and cosine functions

1. Use numpy to build time series data based on sine function
2. Use keras to build a simple regression network, mainly using the LSTM network structure to fit the periodicity of the sine function, and visualize the fitted sine function image and the real function image

#### Related knowledge points

1. Time series data construction and forecasting
2. Time series model building, training, evaluation and visualization based on keras LSTM

```python
# Import necessary libraries

# Build data
import numpy as np

# Build a model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

# Printing progress bar
from tqdm import tqdm

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

#### 1. Construct a data set

This module will use numpy to construct time series data. There are two main steps:

1. Define the sine function (cosine function)
2. Select historical data window size to construct time series data

```python
def ground_func(x):
    """
    sine / cosine function
    Args:
        x: numpy.ndarray
    return:
        sin(x) or cos(x)
    """
    y = np.sin(x)
    return y

def build_data(sequence_data, n_steps):
    """
    Use sine function data to build X, y
    Args:
        sine_data: numpy.ndarray
        n_steps: history data window size
    return:
        X: numpy.ndarray, y: numpy.ndarray
    """

    # init
    X, y = [], []

    seq_len = len(sequence_data)

    for start_idx in tqdm(range(seq_len), total=seq_len):
        end_idx = start_idx + n_steps

        if end_idx >= seq_len:
            break

        cur_x = sequence_data[start_idx: end_idx]
        cur_y = sequence_data[end_idx]

        X.append(cur_x)
        y.append(cur_y)

    X = np.array(X)
    y = np.array(y)

    X = X.reshape(*X.shape, 1)

    return X, y
  
# Construct the original sine/cosine function sequence
xaxis = np.arange(-50 * np.pi, 50 * np.pi, 0.1)
sequence_data = ground_func(xaxis)
len(sequence_data)

# Take 1000 data for visualization
plt.figure(figsize = (20, 8))
plt.plot(xaxis[:1000], sequence_data[:1000])


```

![image-20210901184030073](http://qiniu.hivan.me/MAMTimage-20210901184030073.png?img)



```python
n_steps = 20
X, y = build_data(sequence_data, n_steps)
X.shape, y.shape
"""
 99%|█████████▉| 3122/3142 [00:00<00:00, 1557955.63it/s]
((3122, 20, 1), (3122,))
"""
```

#### 2. Build the model

This module builds a timing model based on the LSTM and Dense layer in keras. The following points need to be noted:
1. Choose the right hidden size
2. Choose a suitable activation function, such as relu, tanh
3. The optimizer chooses sgd, adam, etc.
3. The loss function chooses cross entropy loss function (cross_entropy) or mean square error (mse), etc.

```python
def create_model():
    """
    Build a LSTM model fit sine/cosine function.
    
    hints: 
        1. a LSTM fit time pattern (ref: https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)
        2. a Dense for regression (ref: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)
    """
    model = Sequential()
    model.add(Input(shape = (20, 1)))
    model.add(LSTM(32, activation='tanh'))
    model.add(Dense(1, activation='tanh'))
    model.compile(optimizer = 'adam', loss = 'mse')

    return model

# Initialize the model and print related information
model = create_model()
model.summary()
"""
Instructions for updating:
Call initializer instance with the dtype argument instead of passing it to the constructor
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lstm (LSTM)                  (None, 32)                4352      
_________________________________________________________________
dense (Dense)                (None, 1)                 33        
=================================================================
Total params: 4,385
Trainable params: 4,385
Non-trainable params: 0
_________________________________________________________________
"""

```

#### 3. Model training

```python
# Try to change epochs and add callbacks, such as EarlyStopping (https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping)
history = model.fit(X, y, batch_size = 32, epochs = 25, verbose = 1)
plt.plot(history.history['loss'], label='loss')
plt.legend(loc ='upper right') # draw the loss image
"""
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Epoch 1/25
3122/3122 [==============================] - 4s 1ms/sample - loss: 0.1433
Epoch 2/25
3122/3122 [==============================] - 3s 879us/sample - loss: 0.0072
show more (open the raw output data in a text editor) ...

Epoch 25/25
3122/3122 [==============================] - 3s 858us/sample - loss: 2.2191e-05
"""
```

![image-20210901184239175](http://qiniu.hivan.me/MAMTimage-20210901184239175.png?img)

#### 4. Forecast

This module uses a function different from the training data to construct test data to verify the generalization performance of the model. The main steps are as follows:
1. Define a new function (sine/cosine)
2. Use the trained model to make predictions
3. Visually compare model prediction results with real values

```python
def test_func(x):
    """
    sine/cosine function, different from ground_func above.

    Args:
        x: numpy.ndarray
    return:
        sin(x) or cos(x)
    """
    y = np.cos(x)
    return y
  
test_xaxis = np.arange(0, 10 * np.pi, 0.1)

test_sequence_data = test_func(test_xaxis)

# Use the initial n_steps of historical data to start forecasting, and the subsequent data will use the predicted data as historical data for further forecasting
y_preds = test_sequence_data[:n_steps]

# Step by step forecast
for i in tqdm(range(len(test_xaxis)-n_steps)):
    model_input = y_preds[i: i+n_steps]
    model_input = model_input.reshape((1, n_steps, 1))
    y_pred = model.predict(model_input, verbose = 0)
    y_pred = np.append(y_preds, y_pred)

plt.figure(figsize = (10,8))
plt.plot(test_xaxis[n_steps:], y_preds[n_steps:], label ='predictions')
plt.plot(test_xaxis, test_sequence_data, label ='ground truth')
plt.plot(test_xaxis[:n_steps], y_preds[:n_steps], label ='initial sequence', color ='red')
plt.legend(loc ='upper left')
plt.ylim(-2,2)
plt.show()
"""
100%|██████████| 295/295 [00:01<00:00, 183.91it/s]
"""
```

![image-20210901184333044](http://qiniu.hivan.me/MAMTimage-20210901184333044.png?img)



## Recurrent Neural Networks

[source](https://github.com/hivandu/practise/blob/master/AI_core_competence/Basic%20ability/ex08_Recurrent_Neural_Networks.ipynb) 

```python
import pandas as pd

# load data
timeserise_revenue = pd.read_csv('~/data/course_data/time_serise_revenue.csv')
sales_data = pd.read_csv('~/data/course_data/time_serise_sale.csv')
timeserise_revenue.head()
"""
	Unnamed: 0	day_1	day_2	day_3	day_4	day_5	day_6	day_7	day_8	day_9	...	day_51	day_52	day_53	day_54	day_55	day_56	day_57	day_58	day_59	day_60
0	0	2.622866	2.657832	2.771121	2.815845	2.876267	2.859229	2.844758	2.793797	2.736443	...	1.228701	1.290414	1.474886	1.563295	1.736197	1.797285	1.978940	2.198979	2.277908	2.403300
...
4	4	1.702631	1.825995	2.038047	2.194083	2.313903	2.417883	2.567613	2.650782	2.729691	...	1.258760	1.137150	1.109007	1.104999	1.150137	1.204513	1.221350	1.327023	1.387304	1.557363
5 rows × 61 columns
"""

def sample_from_table(sample_size, dataframe):
    sample_row = dataframe.sample().values[0]

    begin_column = random.randint(0, len(sample_row) - sample_size - 1)

    return (sample_row[begin_column: begin_column + sample_size],
            sample_row[begin_column + 1: begin_column + sample_size + 1])
  
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
import math, random
import matplotlib.pyplot as plt
import seaborn as sns
# Generating a noisy multi-sin wave

class FullyConnected(nn.Module):
    def __init__(self, x_size, hidden_size, output_size):
        super(FullyConnected, self).__init__()
        self.hidden_size = hidden_size

        self.linear_with_tanh = nn.Sequential(
            nn.Linear(10, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, output_size)
        )
    def forward(self, x):
        yhat = self.linear_with_tanh(x)
        return yhat
      
class SimpleRNN(nn.Module):
    def __init__(self, x_size, hidden_size, n_layers, batch_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        # self.inp = nn.Linear(1, hidden_size)
        self.rnn = nn.RNN(x_size, hidden_size, n_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size) # 10 in and 10 out

    def forward(self, inputs, hidden=None):
        hidden = self.__init__hidden()
        # print('Forward hidden {}'.format(hidden.shape))
        # print('Forward inps {}'.format(inputs.shape))
        output, hidden = self.rnn(inputs.float(), hidden.float())
        # print('Out1 {}'.format(output.shape))
        output = self.out(output.float())
        # print('Forward outputs {}'.format(output.shape))

        return output, hidden
    
    def __init__hidden(self):
        hidden = torch.zeros(self.n_layers, self.batch_size, self.hidden_size, dtype = torch.float64)
        return hidden
      
# Set dataset
source_data = sales_data

# Fully Connected Model
n_epochs = 100
n_iters= 50
hidden_size = 2 # try to change this parameters
n_layers = 2
batch_size = 5
seq_length = 10
n_sample_size = 50

x_size = 1

fc_model = FullyConnected(x_size, hidden_size, output_size = seq_length)
fc_model = fc_model.double()

criterion = nn.MSELoss()
optimizer = optim.SGD(fc_model.parameters(), lr = 0.01)

losses = np.zeros(n_epochs)

plt.imshow(fc_model.state_dict()['linear_with_tanh.0.weight'])
plt.show()
```

![image-20210901184546617](http://qiniu.hivan.me/MAMTimage-20210901184546617.png?img)



```python
for epoch in range(n_epochs):
    for iter_ in range(n_iters):
        _inputs, _targets = sample_from_table(n_sample_size, source_data)

        inputs = Variable(torch.from_numpy(np.array([_inputs[0:10],
                                                    _inputs[10:20],
                                                    _inputs[20:30],
                                                    _inputs[30:40],
                                                    _inputs[40:50]],
                                                    dtype = np.double)))

        targets = Variable(torch.from_numpy(np.array([_targets[0:10],
                                                    _targets[10:20],
                                                    _targets[20:30],
                                                    _targets[30:40],
                                                    _targets[40:50]],
                                                    dtype = np.double)))
        
        outputs = fc_model(inputs.double())

        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        losses[epoch] += loss
        if iter_ % 10 == 0:
            plt.clf()
            plt.ion()
            plt.title('Epoch {}, iter {}'.format(epoch, iter_))
            plt.plot(torch.flatten(outputs.detach()), 'r-', linewidth = 1, label = 'Output')
            plt.plot(torch.flatten(targets), 'c-', linewidth = 1, label = 'Label')
            plt.plot(torch.flatten(inputs), 'g-', linewidth = 1, label = 'Input')
            plt.draw()
            plt.pause(0.05)
```

![image-20210901184606015](http://qiniu.hivan.me/MAMTimage-20210901184606015.png?img)

![image-20210901184620638](http://qiniu.hivan.me/MAMTimage-20210901184620638.png?img)

> A total of 5 * 99 pictures were rendered in the middle, so I won’t show them one by one.

![image-20210901184826841](http://qiniu.hivan.me/MAMTimage-20210901184826841.png?img)

### RNN Model

```python
n_epochs = 100
n_iters = 50
hidden_size = 2 # try to change this parameters
n_layers = 2
batch_size = 5
seq_length = 10
n_sample_size = 50

x_size = 1
output_size = 1

rnn_model = SimpleRNN(x_size, hidden_size, n_layers, int(n_sample_size / seq_length), output_size)

criterion = nn.MSELoss()
optimizer = optim.SGD(rnn_model.parameters(), lr = 0.01)

losses = np.zeros(n_epochs)

for epoch in range(n_epochs):
    for iter in range(n_iters):
        _inputs, _targets = sample_from_table(n_sample_size, source_data)

        inputs = Variable(torch.from_numpy(np.array([_inputs[0:10],
                                                    _inputs[10:20],
                                                    _inputs[20:30],
                                                    _inputs[30:40],
                                                    _inputs[40:50]],
                                                    dtype = np.double)).unsqueeze(2))

        targets = Variable(torch.from_numpy(np.array([_targets[0:10],
                                                    _targets[10:20],
                                                    _targets[20:30],
                                                    _targets[30:40],
                                                    _targets[40:50]],
                                                    dtype = np.double)).unsqueeze(2).float())  # [49] 

        # print('Inputs {}, targets {}'.format(inputs.shape, targets.shape))

        # Use teacher forcing 50% of the time
        # force = random.random() < 0.5
        outputs, hidden = rnn_model(inputs.double(), None)

        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        losses[epoch] += loss

        if iter % 10 ==0:
            plt.clf()
            plt.ion()
            plt.title('Epoch {}, iter {}'.format(epoch, iter))
            plt.plot(torch.flatten(outputs.detach()), 'r-', linewidth = 1, label = 'Output')
            plt.plot(torch.flatten(targets), 'c-', linewidth = 1, label = 'Label')
            plt.plot(torch.flatten(inputs), 'g-', linewidth = 1, label = 'Input')
            plt.draw()
            plt.pause(0.05)

# if epoch > 0:
#     print(epoch, loss)


```

![image-20210901184915753](http://qiniu.hivan.me/MAMTimage-20210901184915753.png?img)

> A total of 5 * 99 pictures were rendered in the middle, so I won’t show them one by one.

![image-20210901184958237](http://qiniu.hivan.me/MAMTimage-20210901184958237.png?img)

```python
plt.plot(losses[20:])
plt.show()
```

![image-20210901185025707](http://qiniu.hivan.me/MAMTimage-20210901185025707.png?img)
