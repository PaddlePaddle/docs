.. _cn_api_vision_datasets_FashionMNIST:

FashionMNIST
-------------------------------

.. py:class:: paddle.vision.datasets.FashionMNIST()


    `Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ 数据集

参数
:::::::::
        - image_path (str) - 图像文件路径，如果 ``download`` 设置为 ``True`` ，此参数可以设置为None。默认值为None。
        - label_path (str) - 标签文件路径，如果 ``download`` 设置为 ``True`` ，此参数可以设置为None。默认值为None。
        - mode (str) - ``'train'`` 或 ``'test'`` 模式，默认为 ``'train'`` 。
        - download (bool) - 是否自定下载数据集文件。默认为 ``True`` 。
        - backend (str, optional) - 指定要返回的图像类型：PIL.Image或numpy.ndarray。必须是{'pil'，'cv2'}中的值。如果未设置此选项，将从paddle.vsion.get_image_backend获得这个值。 默认值：None 。

返回
:::::::::

				FashionMNIST数据集实例

代码示例
:::::::::
        
        .. code-block:: python

            from paddle.vision.datasets import FashionMNIST

            mnist = FashionMNIST(mode='test')

            for i in range(len(mnist)):
                sample = mnist[i]
                print(sample[0].size, sample[1])

    
