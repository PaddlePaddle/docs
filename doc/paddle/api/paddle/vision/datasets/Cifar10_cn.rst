.. _cn_api_vision_datasets_Cifar10:

Cifar10
-------------------------------

.. py:class:: paddle.vision.datasets.Cifar10()


    Implementation of `Cifar-10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_
    dataset, which has 10 categories.

    参数
:::::::::
        data_file(str): path to data file, can be set None if
            :attr:`download` is True. Default None
        mode(str): 'train', 'test' mode. Default 'train'.
        transform(callable): transform to perform on image, None for on transform.
        download(bool): whether to download dataset automatically if
            :attr:`data_file` is not set. Default True

    Returns:
        Dataset: instance of cifar-10 dataset

    代码示例
:::::::::

        .. code-block:: python

            import paddle
            import paddle.nn as nn
            from paddle.vision.datasets import Cifar10
            from paddle.vision.transforms import Normalize

            class SimpleNet(paddle.nn.Layer):
                def __init__(self):
                    super(SimpleNet, self).__init__()
                    self.fc = nn.Sequential(
                        nn.Linear(3072, 10),
                        nn.Softmax())

                def forward(self, image, label):
                    image = paddle.reshape(image, (3, -1))
                    return self.fc(image), label

            paddle.disable_static()

            normalize = Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])
            cifar10 = Cifar10(mode='train', transform=normalize)

            for i in range(10):
                image, label = cifar10[i]
                image = paddle.to_tensor(image)
                label = paddle.to_tensor(label)

                model = SimpleNet()
                image, label = model(image, label)
                print(image.numpy().shape, label.numpy().shape)

    