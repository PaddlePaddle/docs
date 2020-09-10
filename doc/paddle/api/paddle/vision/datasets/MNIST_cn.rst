.. _cn_api_vision_datasets_MNIST:

MNIST
-------------------------------

.. py:class:: paddle.vision.datasets.MNIST()


    Implementation of `MNIST <http://yann.lecun.com/exdb/mnist/>`_ dataset

    参数
:::::::::
        image_path(str): path to image file, can be set None if
            :attr:`download` is True. Default None
        label_path(str): path to label file, can be set None if
            :attr:`download` is True. Default None
        chw_format(bool): If set True, the output shape is [1, 28, 28],
            otherwise, output shape is [1, 784]. Default True.
        mode(str): 'train' or 'test' mode. Default 'train'.
        download(bool): whether to download dataset automatically if
            :attr:`image_path` :attr:`label_path` is not set. Default True

    Returns:
        Dataset: MNIST Dataset.

    代码示例
:::::::::
        
        .. code-block:: python

            from paddle.vision.datasets import MNIST

            mnist = MNIST(mode='test')

            for i in range(len(mnist)):
                sample = mnist[i]
                print(sample[0].shape, sample[1])

    