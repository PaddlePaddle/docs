.. _cn_doc_data_preprocessing:

数据预处理
================

训练过程中有时会遇到过拟合的问题，其中一个解决方法就是对训练数据做增强，对数据进行处理得到不同的图像，从而泛化数据集。数据增强API是定义在领域目录的transofrms下，这里我们介绍两种使用方式，一种是基于框架内置数据集，一种是基于自己定义的数据集。


飞桨框架内置数据集
-----------------------

针对飞桨框架内置图像数据集的预处理，飞桨框架将这部分API整合到\ ``paddle.vision.transforms``\ 下，具体包括如下方法：

.. code:: ipython3

    import paddle
    print('数据处理方法：', paddle.vision.transforms.__all__)


.. parsed-literal::

    数据处理方法： ['BaseTransform', 'Compose', 'Resize', 'RandomResizedCrop', 'CenterCrop', 'RandomHorizontalFlip', 'RandomVerticalFlip', 'Transpose', 'Normalize', 'BrightnessTransform', 'SaturationTransform', 'ContrastTransform', 'HueTransform', 'ColorJitter', 'RandomCrop', 'Pad', 'RandomRotation', 'Grayscale', 'ToTensor', 'to_tensor', 'hflip', 'vflip', 'resize', 'pad', 'rotate', 'to_grayscale', 'crop', 'center_crop', 'adjust_brightness', 'adjust_contrast', 'adjust_hue', 'normalize'

这里，我们随机调整图像的亮度、对比度、饱和度，并调整图像的大小，对图像的其他调整，可以参考具体的API文档。

.. code:: ipython3

    from paddle.vision.transforms import Compose, Resize, ColorJitter


    # 定义想要使用那些数据增强方式，这里用到了随机调整亮度、对比度和饱和度，改变图片大小
    transform = Compose([ColorJitter(), Resize(size=100)])

    # 通过transform参数传递定义好的数据增项方法即可完成对自带数据集的应用
    train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)


自定义数据集
-----------------------

针对自定义数据集使用数据增强有两种方式，一种是在数据集的构造函数中进行数据增强方法的定义，之后对__getitem__中返回的数据进行应用。另外一种方式也可以给自定义的数据集类暴漏一个构造参数，在实例化类的时候将数据增强方法传递进去。

.. code:: ipython3

    import paddle
    from paddle.io import Dataset


    class MyDataset(Dataset):
        def __init__(self, mode='train'):
            super(MyDataset, self).__init__()

            if mode == 'train':
                self.data = [
                    ['traindata1', 'label1'],
                    ['traindata2', 'label2'],
                    ['traindata3', 'label3'],
                    ['traindata4', 'label4'],
                ]
            else:
                self.data = [
                    ['testdata1', 'label1'],
                    ['testdata2', 'label2'],
                    ['testdata3', 'label3'],
                    ['testdata4', 'label4'],
                ]

            # 定义要使用的数据预处理方法，针对图片的操作
            self.transform = Compose([ColorJitter(), Resize(size=100)])

        def __getitem__(self, index):
            data = self.data[index][0]

            # 在这里对训练数据进行应用
            # 这里只是一个示例，测试时需要将数据集更换为图片数据进行测试
            data = self.transform(data)

            label = self.data[index][1]

            return data, label

        def __len__(self):
            return len(self.data)
