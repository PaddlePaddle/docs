# Dataset and DataLoader

Deep learning models require a large amount of data to train the model, and this process is all numerical computation and cannot directly use images and text. Therefore, it is necessary to process the original data files and convert them into `Tensor` that can be used by models.

## 1. Datasets in Paddle

Paddle provides common datasets in `paddle.vision.datasets` and `paddle.text`, you can see the details with the following code.


```python
import paddle
print('CV datasets: ', paddle.vision.datasets.__all__)
print('NLP datasets: ', paddle.text.__all__)
```

    CV datasets:  ['DatasetFolder', 'ImageFolder', 'MNIST', 'FashionMNIST', 'Flowers', 'Cifar10', 'Cifar100', 'VOC2012']
    NLP datasets:  ['Conll05st', 'Imdb', 'Imikolov', 'Movielens', 'UCIHousing', 'WMT14', 'WMT16', 'ViterbiDecoder', 'viterbi_decode']


You can load the MNIST dataset in the following code, use `mode` to identify the training datasets and validation dataset. The dataset will automatically download the local cache directory ~/.cache/paddle/dataset.


```python
from paddle.vision.transforms import ToTensor
# training datasets
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=ToTensor())

# validation datasets
val_dataset = paddle.vision.datasets.MNIST(mode='test', transform=ToTensor())
```

## 2. Creating a Custom Dataset

In general, you need to create a custom datasets for your own data. You can use `paddle.io.Dataset`in Paddle to quickly implement.


```python
import paddle
from paddle.io import Dataset

BATCH_SIZE = 64
BATCH_NUM = 20

IMAGE_SIZE = (28, 28)
CLASS_NUM = 10


class MyDataset(Dataset):
    """
    step1: inherit paddle.io.Dataset
    """
    def __init__(self, num_samples):
        """
        step2: implement the constructor function
        """
        super(MyDataset, self).__init__()
        self.num_samples = num_samples

    def __getitem__(self, index):
        """
        step3: define how to get data when specifying index 
        and return a data (training data, label)
        """
        data = paddle.uniform(IMAGE_SIZE, dtype='float32')
        label = paddle.randint(0, CLASS_NUM-1, dtype='int64')

        return data, label

    def __len__(self):
        """
        step4: return the numbers of dataset
        """
        return self.num_samples

# test
custom_dataset = MyDataset(BATCH_SIZE * BATCH_NUM)

print('=============custom dataset=============')
for data, label in custom_dataset:
    print(data.shape, label.shape)
    break
```

    =============custom dataset=============
    [28, 28] [1]


Based on the above approach, you can create your custom dataset according to the actual scenario.

## 3. DataLoader

Paddle recommends using `paddle.io.DataLoader` to complete the data loading. A simple example is as follows.


```python
train_loader = paddle.io.DataLoader(custom_dataset, batch_size=BATCH_SIZE, shuffle=True)
# if load paddle provides dataset provides, replace custom_dataset with train_dataset
for batch_id, data in enumerate(train_loader()):
    x_data = data[0]
    y_data = data[1]

    print(x_data.shape)
    print(y_data.shape)
    break
```

    [64, 28, 28]
    [64, 1]


With the above method, you define a data iterator `train_loader`, which is used to load the training data. The batch size of dataset is set to 64 by batch_size=64, and the data is scrambled before loading by shuffle=True. In addition, you can ues multi-process data loading by setting num_workers to improve loading speed.
