.. _cn_doc_data_load:

数据集定义与加载
================

深度学习模型在训练时需要大量的数据来完成模型调优，这个过程均是数字的计算，无法直接使用原始图片和文本等来完成计算。因此与需要对原始的各种数据文件进行处理，转换成深度学习模型可以使用的数据类型。

一、框架自带数据集
---------------------

飞桨框架将深度学习任务中常用到的数据集作为领域API开放，对应API所在目录为\ ``paddle.vision.datasets``\ 与\ ``paddle.text``\，你可以通过以下代码飞桨框架中提供了哪些数据集。

.. code:: ipython3
    
    import paddle
    print('视觉相关数据集：', paddle.vision.datasets.__all__)
    print('自然语言相关数据集：', paddle.text.__all__)


.. parsed-literal::

    视觉相关数据集： ['DatasetFolder', 'ImageFolder', 'MNIST', 'FashionMNIST', 'Flowers', 'Cifar10', 'Cifar100', 'VOC2012']
    自然语言相关数据集： ['Conll05st', 'Imdb', 'Imikolov', 'Movielens', 'UCIHousing', 'WMT14', 'WMT16']

.. warning::
    除\ ``paddle.vision.dataset``\ 与\ ``paddle.text``\ 外，飞桨框架还内置了另一套数据集，路径为\ ``paddle.dataset.*``\ ，但是该数据集的使用方式较老，会在未来的版本废弃，请尽量不要使用该目录下数据集的API。

这里你可以定义手写数字体的数据集，其他数据集的使用方式也都类似。用\ ``mode``\ 来标识训练集与测试集。数据集接口会自动从远端下载数据集到本机缓存目录\ ``~/.cache/paddle/dataset``\ 。

.. code:: ipython3

    from paddle.vision.transforms import ToTensor
    # 训练数据集 用ToTensor将数据格式转为Tensor
    train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=ToTensor())

    # 验证数据集
    val_dataset = paddle.vision.datasets.MNIST(mode='test', transform=ToTensor())


二、自定义数据集
-------------------

在实际的场景中，更多需要使用你已有的相关数据来定义数据集。你可以使用飞桨提供的\ ``paddle.io.Dataset``\ 基类，来快速实现自定义数据集。

.. code:: ipython3

    import paddle
    from paddle.io import Dataset

    BATCH_SIZE = 64
    BATCH_NUM = 20

    IMAGE_SIZE = (28, 28)
    CLASS_NUM = 10


    class MyDataset(Dataset):
        """
        步骤一：继承paddle.io.Dataset类
        """
        def __init__(self, num_samples):
            """
            步骤二：实现构造函数，定义数据集大小
            """
            super(MyDataset, self).__init__()
            self.num_samples = num_samples
        
        def __getitem__(self, index):
            """
            步骤三：实现__getitem__方法，定义指定index时如何获取数据，并返回单条数据（训练数据，对应的标签）
            """
            data = paddle.uniform(IMAGE_SIZE, dtype='float32')
            label = paddle.randint(0, CLASS_NUM-1, dtype='int64')

            return data, label

        def __len__(self):
            """
            步骤四：实现__len__方法，返回数据集总数目
            """
            return self.num_samples

    # 测试定义的数据集
    custom_dataset = MyDataset(BATCH_SIZE * BATCH_NUM)

    print('=============custom dataset=============')
    for data, label in custom_dataset:
        print(data.shape, label.shape)
        break


.. parsed-literal::

    =============custom dataset=============
    [28, 28] [1]

通过以上的方式，你就可以根据实际场景，构造自己的数据集。


三、数据加载
------------

飞桨推荐使用\ ``paddle.io.DataLoader``\ 完成数据的加载。简单的示例如下：

.. code:: ipython3

    train_loader = paddle.io.DataLoader(custom_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # 如果要加载内置数据集，将 custom_dataset 换为 train_dataset 即可
    for batch_id, data in enumerate(train_loader()):
        x_data = data[0]
        y_data = data[1]

        print(x_data.shape)
        print(y_data.shape)
        break

.. parsed-literal::

    [64, 28, 28]
    [64, 1]

通过上述的方法，你就定义了一个数据迭代器\ ``train_loader``\ , 用于加载训练数据。通过\ ``batch_size=64``\ 设置了数据集的批大小为64，通过\ ``shuffle=True``\ ，在取数据前会打乱数据。此外，你还可以通过设置\ ``num_workers``\ 来开启多进程数据加载，提升加载速度。

.. note::
    DataLoader 默认用异步加载数据的方式来读取数据，一方面可以提升数据加载的速度，另一方面也会占据更少的内存。如果你需要同时加载全部数据到内存中，请设置\ ``use_buffer_reader=False``\ 。
