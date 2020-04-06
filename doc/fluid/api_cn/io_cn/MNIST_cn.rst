.. _cn_api_fluid_io_MNIST:

MNIST
-------------------------------

.. py:class:: paddle.fluid.io.MNIST(image_path=None, label_path=None, mode='train', download=True)

MNIST数据集

参数:
    - **image_path** (str) - 数据集图像文件路径，若 ``download`` 为True， ``image_path`` 可设置为None。默认值为None。
    - **label_path** (str) - 数据集标注文件路径，若 ``download`` 为True， ``label_path`` 可设置为None。默认值为None。
    - **mode** (str) - 数据集模式，即读取 ``'train'`` 或者 ``'test'`` 数据。默认值为 ``'train'`` 。
    - **download** (bool) - 当 ``image_path`` 或 ``label_path`` 为None时，是否自动下载数据集。默认值为True。

返回：MNIST数据集

返回类型: Dataset

**代码示例**

.. code-block:: python

    from paddle.fluid.io import MNIST

    mnist = MNIST(mode='test')

    for i in range(len(mnist)):
        sample = mnist[i]
        print(sample[0].shape, sample[1])

