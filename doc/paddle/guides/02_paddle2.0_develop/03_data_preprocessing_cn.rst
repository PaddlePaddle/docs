.. _cn_doc_data_preprocessing:

数据预处理
================

训练过程中有时会遇到过拟合的问题，其中一个解决方法就是对训练数据做增强，对数据进行处理得到不同的图像，从而泛化数据集。数据增强API是定义在领域目录的transofrms下，这里介绍两种使用方式，一种是基于框架内置数据集，一种是基于自定义的数据集。


一、飞桨框架内置数据集
-----------------------

针对飞桨框架内置图像数据集的预处理，飞桨框架将这部分API整合到\ ``paddle.vision.transforms``\ 下，你可以通过以下方式查看：

.. code:: ipython3

    import paddle
    print('数据处理方法：', paddle.vision.transforms.__all__)


.. parsed-literal::

    数据处理方法： ['BaseTransform', 'Compose', 'Resize', 'RandomResizedCrop', 'CenterCrop', 'RandomHorizontalFlip', 'RandomVerticalFlip', 'Transpose', 'Normalize', 'BrightnessTransform', 'SaturationTransform', 'ContrastTransform', 'HueTransform', 'ColorJitter', 'RandomCrop', 'Pad', 'RandomRotation', 'Grayscale', 'ToTensor', 'to_tensor', 'hflip', 'vflip', 'resize', 'pad', 'rotate', 'to_grayscale', 'crop', 'center_crop', 'adjust_brightness', 'adjust_contrast', 'adjust_hue', 'normalize']

你可以同构以下方式随机调整图像的亮度、对比度、饱和度，并调整图像的大小，对图像的其他调整，可以参考相关的API文档。

.. code:: ipython3

    from paddle.vision.transforms import Compose, Resize, ColorJitter

    # 定义想要使用的数据增强方式，这里包括随机调整亮度、对比度和饱和度，改变图片大小
    transform = Compose([ColorJitter(), Resize(size=32)])

    # 通过transform参数传递定义好的数据增强方法即可完成对自带数据集的增强
    train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)


二、自定义数据集
-----------------------

对于自定义的数据集，你可以在数据集的构造函数中进行数据增强方法的定义，之后对 ``__getitem__`` 中返回的数据进行应用，就可以完成自定义数据增强。

.. code:: ipython3

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
            # 在 `__init__` 中定义数据增强方法，此处为调整图像大小
            self.transform = Compose([Resize(size=32)])
        
        def __getitem__(self, index):
            data = paddle.uniform(IMAGE_SIZE, dtype='float32')
            # 在 `__getitem__` 中对数据集使用数据增强方法
            data = self.transform(data.numpy())

            label = paddle.randint(0, CLASS_NUM-1, dtype='int64')

            return data, label

        def __len__(self):
            return self.num_samples

    # 测试定义的数据集
    custom_dataset = MyDataset(BATCH_SIZE * BATCH_NUM)

    print('=============custom dataset=============')
    for data, label in custom_dataset:
        print(data.shape, label.shape)
        break


.. parsed-literal::

    =============custom dataset=============
    [32, 32] [1]


可以看出，输出的形状从 ``[28, 28, 1]`` 变为了 ``[32, 32, 1]``，证明完成了图像的大小调整。