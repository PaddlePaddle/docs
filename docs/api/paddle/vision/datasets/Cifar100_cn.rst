.. _cn_api_paddle_vision_datasets_Cifar100:

Cifar100
-------------------------------

.. py:class:: paddle.vision.datasets.Cifar100(data_file=None, mode='train', transform=None, download=True, backend=None)


`Cifar-100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ 数据集的实现，包含 100 种类别。

参数
:::::::::

  - **data_file** (str，可选) - 数据集文件路径，如果 ``download`` 参数设置为 ``True``，``data_file`` 参数可以设置为 ``None``。默认值为 ``None``，默认存放在：``~/.cache/paddle/dataset/cifar``。
  - **mode** (str，可选) - ``'train'`` 或 ``'test'`` 模式两者之一，默认值为 ``'train'``。
  - **transform** (Callable，可选) - 图片数据的预处理，若为 ``None`` 即为不做预处理。默认值为 ``None``。
  - **download** (bool，可选) - 当 ``data_file`` 是 ``None`` 时，该参数决定是否自动下载数据集文件。默认值为 ``True``。
  - **backend** (str，可选) - 指定要返回的图像类型：PIL.Image 或 numpy.ndarray。必须是 {'pil'，'cv2'} 中的值。如果未设置此选项，将从 :ref:`paddle.vision.get_image_backend <cn_api_paddle_vision_get_image_backend>` 获得这个值。默认值为 ``None``。

返回
:::::::::

:ref:`cn_api_paddle_io_Dataset`，Cifar100 数据集实例。

代码示例
:::::::::

COPY-FROM: paddle.vision.datasets.Cifar100
