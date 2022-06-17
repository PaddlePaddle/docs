.. _cn_api_vision_datasets_Cifar100:

Cifar100
-------------------------------

.. py:class:: paddle.vision.datasets.Cifar100()


    `Cifar-100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ 数据集的实现，数据集包含100种类别。

参数
:::::::::
        - data_file (str) - 数据集文件路径，如果 ``download`` 参数设置为 ``True`` ， ``data_file`` 参数可以设置为 ``None``。默认值为 ``None``，默认存放在：``~/.cache/paddle/dataset/cifar``
        - mode (str) - ``'train'`` 或 ``'test'`` 模式，默认为 ``'train'`` 。
        - transform (callable) - 图片数据的预处理，若为 ``None`` 即为不做预处理。默认值为 ``None``。
        - download (bool) - 当 ``data_file`` 是 ``None`` 时，该参数决定是否自动下载数据集文件。默认为 ``True`` 。
        - backend (str，optional) - 指定要返回的图像类型：PIL.Image或numpy.ndarray。必须是{'pil'，'cv2'}中的值。如果未设置此选项，将从paddle.vsion.get_image_backend获得这个值。默认值：``None`` 。
返回
:::::::::

				Cifar100数据集实例。

代码示例
:::::::::

COPY-FROM: paddle.vision.datasets.Cifar100