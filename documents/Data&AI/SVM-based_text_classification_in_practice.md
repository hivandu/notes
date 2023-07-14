# SVM-based Text Classification in Practice

> The source code: [SVM-based Text Classification in Practice](https://github.com/hivandu/Colab/blob/master/AI_data/SVM-based%20text%20classification%20in%20practice.ipynb)

> 'cnews.train.txt' data cannot be uploaded because it is too large, so it needs to be decompressed and imported after compression.


Use SVM to implement a simple text classification based on bag of words and support vector machine.

## import data

```python
# import
import codecs
import os
import jieba
```

Chinese news data is prepared as a sample data set. The number of training data is 50,000 and the number of test data is 10,000. All data is divided into 10 categories: sports, finance, real estate, home furnishing, education, technology, fashion, current affairs, games and entertainment . From the training text, you can load the code, view the data format and samples:

```python

data_train = './data/cnews.train.txt' # training data file name  
data_test = './data/cnews.test.txt'  # test data file name
vocab = './data/cnews.vocab.txt' # dictionary

with codecs.open(data_train, 'r', 'utf-8') as f:
    lines = f.readlines()

# print sample content
label, content = lines[0].strip('\r\n').split('\t')
content
```

Take the first item of the training data as an example to segment the loaded news data. Here I use the word segmentation function of LTP, you can also use jieba, and the segmentation results are displayed separated by "/" symbols.

```python
# print word segment results
segment = jieba.cut(content)
print('/'.join(segment))
```

To sort out the above logic a bit, implement a class to load training and test data and perform word segmentation.

```python
# cut data
def process_line(idx, line):
    data = tuple(line.strip('\r\n').split('\t'))
    if not len(data)==2:
        return None
    content_segged = list(jieba.cut(data[1]))
    if idx % 1000 == 0:
        print('line number: {}'.format(idx))
    return (data[0], content_segged)
    
# data loading method
def load_data(file):
    with codecs.open(file, 'r', 'utf-8') as f:
        lines = f.readlines()
    data_records = [process_line(idx, line) for idx, line in enumerate(lines)]
    data_records = [data for data in data_records if data is not None]
    return data_records

# load and process training data
train_data = load_data(data_train)
print('first training data: label {} segment {}'.format(train_data[0][0], '/'.join(train_data[0][1])))
# load and process testing data
test_data = load_data(data_test)
print('first testing data: label {} segment {}'.format(test_data[0][0], '/'.join(test_data[0][1])))
```

After spending some time on word segmentation, you can start building a dictionary. The dictionary is built from the training set and sorted by word frequency.

```python
def build_vocab(train_data, thresh):
    vocab = {'<UNK>': 0}
    word_count = {} # word frequency
    for idx, data in enumerate(train_data):
        content = data[1]
        for word in content:
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1
    word_list = [(k, v) for k, v in word_count.items()]
    print('word list length: {}'.format(len(word_list)))
    word_list.sort(key = lambda x : x[1], reverse = True) # sorted by word frequency
    word_list_filtered = [word for word in word_list if word[1] > thresh]
    print('word list length after filtering: {}'.format(len(word_list_filtered)))
    # construct vocab
    for word in word_list_filtered:
        vocab[word[0]] = len(vocab)
    print('vocab size: {}'.format(len(vocab))) # vocab size is word list size +1 due to unk token
    return vocab

vocab = build_vocab(train_data, 1)
```

In addition, according to category, we know that the label itself also has a "dictionary":

```python
def build_label_vocab(cate_file):
    label_vocab = {}
    with codecs.open(cate_file, 'r', 'utf-8') as f:
        for lines in f:
            line = lines.strip().split('\t')
            label_vocab[line[0]]  = int(line[1])
    return label_vocab

label_vocab = build_label_vocab('./data/cnews.category.txt')
print(f'label vocab: {label_vocab}')
```


Next, construct the id-based training and test sets, because we only consider the bag of words, so the order of words is excluded. Constructed to look like libsvm can eat. Note that because the bag of word model

```python
def construct_trainable_matrix(corpus, vocab, label_vocab, out_file):
    records = []
    for idx, data in enumerate(corpus):
        if idx % 1000 == 0:
            print('process {} data'.format(idx))
        label = str(label_vocab[data[0]]) # label id
        token_dict = {}
        for token in data[1]:
            token_id = vocab.get(token, 0)
            if token_id in token_dict:
                token_dict[token_id] += 1
            else:
                token_dict[token_id] = 1
        feature = [str(int(k) + 1) + ':' + str(v) for k,v in token_dict.items()]
        feature_text = ' '.join(feature)
        records.append(label + ' ' + feature_text)
    
    with open(out_file, 'w') as f:
        f.write('\n'.join(records))

construct_trainable_matrix(train_data, vocab, label_vocab, './data/train.svm.txt')
construct_trainable_matrix(test_data, vocab, label_vocab, './data/test.svm.txt')
```

## Training process

The remaining core model is simple: use libsvm to train the support vector machine, let your svm eat the training and test files you have processed, and then use the existing method of libsvm to train, we can change different parameter settings . The documentation of libsvm can be viewed [here](https://www.csie.ntu.edu.tw/~cjlin/libsvm/), where the "-s, -t, -c" parameters are more important, and they decide what you choose Svm, your choice of kernel function, and your penalty coefficient.

```python
from libsvm import svm
from libsvm.svmutil import svm_read_problem,svm_train,svm_predict,svm_save_model,svm_load_model

# train svm
train_label, train_feature = svm_read_problem('./data/train.svm.txt')
print(train_label[0], train_feature[0])
model=svm_train(train_label,train_feature,'-s 0 -c 5 -t 0 -g 0.5 -e 0.1')

# predict
test_label, test_feature = svm_read_problem('./data/test.svm.txt')
print(test_label[0], test_feature[0])
p_labs, p_acc, p_vals = svm_predict(test_label, test_feature, model)

print('accuracy: {}'.format(p_acc))
```

After a period of training, we can observe the experimental results. You can change different svm types, penalty coefficients, and kernel functions to optimize the results.