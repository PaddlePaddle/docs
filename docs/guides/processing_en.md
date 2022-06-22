# Processing

Overfitting sometimes happened in training, and one of the solutions is to do data preprocessing on the training data. The data transform API is in `paddle.vision.transofrms.*`. This tutorial introduces two ways to use it, one is based dataset Paddle provided and the other is a custom dataset.

## 1. Dataset Paddle provided

Data processing API of Paddle in `paddle.vision.transforms`. You can view the APIs by the following code.


```python
import paddle

print('data processing functions: ', paddle.vision.transforms.__all__)
```

    data processing functions:  ['BaseTransform', 'Compose', 'Resize', 'RandomResizedCrop', 'CenterCrop', 'RandomHorizontalFlip', 'RandomVerticalFlip', 'Transpose', 'Normalize', 'BrightnessTransform', 'SaturationTransform', 'ContrastTransform', 'HueTransform', 'ColorJitter', 'RandomCrop', 'Pad', 'RandomRotation', 'Grayscale', 'ToTensor', 'to_tensor', 'hflip', 'vflip', 'resize', 'pad', 'rotate', 'to_grayscale', 'crop', 'center_crop', 'adjust_brightness', 'adjust_contrast', 'adjust_hue', 'normalize']


You can randomly adjust the brightness, contrast and saturation of the image and resize in the following ways. For other adjustments, you can refer to the relevant API documentation.


```python
from paddle.vision.transforms import Compose, Resize, ColorJitter

# define data processing function
transform = Compose([ColorJitter(), Resize(size=32)])

train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
```

## 2. Custom Dataset

For custom datasets, you can define the data processing function in the constructor of the dataset, and later apply it to the data returned in `__getitem__`.


```python
import paddle
from paddle.io import Dataset
from paddle.vision.transforms import Compose, Resize

BATCH_SIZE = 64
BATCH_NUM = 20

IMAGE_SIZE = (28, 28)
CLASS_NUM = 10

class MyDataset(Dataset):
    def __init__(self, num_samples):
        super(MyDataset, self).__init__()
        self.num_samples = num_samples
        #  define processing function, here is resize
        self.transform = Compose([Resize(size=32)])

    def __getitem__(self, index):
        data = paddle.uniform(IMAGE_SIZE, dtype='float32')
        # apply processing function to data
        data = self.transform(data.numpy())

        label = paddle.randint(0, CLASS_NUM-1, dtype='int64')

        return data, label

    def __len__(self):
        return self.num_samples

# test
custom_dataset = MyDataset(BATCH_SIZE * BATCH_NUM)

print('=============custom dataset=============')
for data, label in custom_dataset:
    print(data.shape, label.shape)
    break
```

    =============custom dataset=============


    W0105 00:19:13.706831    99 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.0, Runtime API Version: 10.1
    W0105 00:19:13.710531    99 device_context.cc:465] device: 0, cuDNN Version: 7.6.


    (32, 32) [1]


It can be seen that the output shape has changed from [28, 28, 1] to [32, 32, 1], which proves that the resizing of the image has been completed.
