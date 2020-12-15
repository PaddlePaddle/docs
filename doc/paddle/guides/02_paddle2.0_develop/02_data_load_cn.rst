.. _cn_doc_data_load:

数据集定义与加载
================

对于深度学习任务，均是框架针对各种类型数字的计算，是无法直接使用原始图片和文本等文件来完成。那么就是涉及到了一项动作，就是将原始的各种数据文件进行处理加工，转换成深度学习任务可以使用的数据。

框架自带数据集
---------------------

飞桨框架将一些我们常用到的数据集作为领域API对用户进行开放，对应API所在目录为\ ``paddle.vision.datasets``\ 与\ ``paddle.text.datasets``\，那么我们先看下提供了哪些数据集。

.. code:: ipython3

    print('视觉相关数据集：', paddle.vision.datasets.__all__)
    print('自然语言相关数据集：', paddle.text.datasets.__all__)


.. parsed-literal::

    视觉相关数据集： ['DatasetFolder', 'ImageFolder', 'MNIST', 'FashionMNIST', 'Flowers', 'Cifar10', 'Cifar100', 'VOC2012']
    自然语言相关数据集： ['Conll05st', 'Imdb', 'Imikolov', 'Movielens', 'UCIHousing', 'WMT14', 'WMT16']

.. warning::
    除\ ``paddle.vision.dataset``\ 与\ ``paddle.text.dataset``\ 外，飞桨框架还内置了另一套数据集，路径为\ ``paddle.dataset.*``\ ，但是该数据集的使用方式较老，会在未来的版本废弃，请您尽量不要使用该目录下数据集的API。

这里我们加载一个手写数字识别的数据集，其他数据集的使用方式也都类似。用\ ``mode``\ 来标识训练集与测试集。数据集接口会自动从远端下载数据集到本机缓存目录\ ``~/.cache/paddle/dataset``\ 。

.. code:: ipython3

    from paddle.vision.transforms import ToTensor
    # 训练数据集 用ToTensor将数据格式转为Tensor
    train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=ToTensor())

    # 验证数据集
    val_dataset = paddle.vision.datasets.MNIST(mode='test', transform=ToTensor())


自定义数据集
-------------------

在实际的使用场景中，更多的时候我们是需要自己使用已有的相关数据来定义数据集，那么这里我们通过一个案例来了解如何进行数据集的定义，飞桨为用户提供了\ ``paddle.io.Dataset``\ 基类，让用户通过类的集成来快速实现数据集定义。

.. code:: ipython3

    from paddle.io import Dataset


    class MyDataset(Dataset):
        """
        步骤一：继承paddle.io.Dataset类
        """
        def __init__(self, mode='train'):
            """
            步骤二：实现构造函数，定义数据读取方式，划分训练和测试数据集
            """
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

        def __getitem__(self, index):
            """
            步骤三：实现__getitem__方法，定义指定index时如何获取数据，并返回单条数据（训练数据，对应的标签）
            """
            data = self.data[index][0]
            label = self.data[index][1]

            return data, label

        def __len__(self):
            """
            步骤四：实现__len__方法，返回数据集总数目
            """
            return len(self.data)

    # 测试定义的数据集
    train_dataset2 = MyDataset(mode='train')
    val_dataset2 = MyDataset(mode='test')

    print('=============train dataset=============')
    for data, label in train_dataset2:
        print(data, label)

    print('=============evaluation dataset=============')
    for data, label in val_dataset2:
        print(data, label)


.. parsed-literal::

    =============train dataset=============
    traindata1 label1
    traindata2 label2
    traindata3 label3
    traindata4 label4
    =============evaluation dataset=============
    testdata1 label1
    testdata2 label2
    testdata3 label3
    testdata4 label4

通过以上的方式，就可以根据实际场景，构造自己的数据集。


数据加载
------------

飞桨推荐使用\ ``paddle.io.DataLoader``\ 完成数据的加载。简单的示例如下：

.. code:: ipython3

    train_loader = paddle.io.DataLoader(train_dataset, batch_size=64, shuffle=True)
    for batch_id, data in enumerate(train_loader()):
        x_data = data[0]
        y_data = data[1]

        print(x_data.numpy().shape)
        print(y_data.numpy().shape)

.. parsed-literal::

    (64, 1, 28, 28)
    (64, 1)

通过上述的方法，我们定义了一个数据迭代器\ ``train_loader``\ , 用于加载训练数据。通过\ ``batch_size=64``\ 我们设置了数据集的批大小为64，通过\ ``shuffle=True``\ ，我们在取数据前会打乱数据。此外，我们还可以通过设置\ ``num_workers``\ 来开启多进程数据加载，提升加载速度。

.. note::
    DataLoader 默认用异步加载数据的方式来读取数据，一方面可以提升数据加载的速度，另一方面也会占据更少的内存。如您需要同时加载全部数据到内存中，请设置\ ``use_buffer_reader=False``\ 。
