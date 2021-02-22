.. _cn_api_vision_datasets_MNIST:

MNIST
-------------------------------

.. py:class:: paddle.vision.datasets.MNIST()


    `MNIST <http://yann.lecun.com/exdb/mnist/>`_ 数据集

参数
:::::::::
        - image_path (str) - 图像文件路径，如果 ``download`` 参数设置为 ``True`` ， ``image_path`` 参数可以设置为 ``None`` 。默认值为 ``None`` 。
        - label_path (str) - 标签文件路径，如果 ``download`` 参数设置为 ``True`` ， ``label_path`` 参数可以设置为 ``None`` 。默认值为 ``None`` 。
        - chw_format (bool) - 若为 ``True`` 输出形状为 ``[1, 28, 28]`` , 否则为 ``[1, 784]`` 。默认值为 ``True`` 。
        - mode (str) - ``'train'`` 或 ``'test'`` 模式，默认为 ``'train'`` 。
        - download (bool) - 当 ``data_file`` 是 ``None`` 时，该参数决定是否自动下载数据集文件。默认为 ``True`` 。

返回
:::::::::

				MNIST数据集实例

代码示例
:::::::::
        
        .. code-block:: python

            from paddle.vision.datasets import MNIST

            mnist = MNIST(mode='test')

            for i in range(len(mnist)):
                sample = mnist[i]
                print(sample[0].shape, sample[1])

    
