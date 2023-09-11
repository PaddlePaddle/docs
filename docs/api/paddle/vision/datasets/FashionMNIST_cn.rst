.. _cn_api_paddle_vision_datasets_FashionMNIST:

FashionMNIST
-------------------------------

.. py:class:: paddle.vision.datasets.FashionMNIST(image_path=None, label_path=None, mode='train', transform=None, download=True, backend=None)


`Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ 数据集的实现。

参数
:::::::::

  - **image_path** (str，可选) - 图像文件路径，如果 ``download`` 参数设置为 ``True``，``image_path`` 参数可以设置为 ``None``。默认值为 ``None``，默认存放在：``~/.cache/paddle/dataset/fashion-mnist``。
  - **label_path** (str，可选) - 标签文件路径，如果 ``download`` 参数设置为 ``True``，``label_path`` 参数可以设置为 ``None``。默认值为 ``None``，默认存放在：``~/.cache/paddle/dataset/fashion-mnist``。
  - **mode** (str，可选) - ``'train'`` 或 ``'test'`` 模式两者之一，默认值为 ``'train'``。
  - **transform** (Callable，可选) - 图片数据的预处理，若为 ``None`` 即为不做预处理。默认值为 ``None``。
  - **download** (bool，可选) - 当 ``data_file`` 是 ``None`` 时，该参数决定是否自动下载数据集文件。默认值为  ``True``。
  - **backend** (str，可选) - 指定要返回的图像类型：PIL.Image 或 numpy.ndarray。必须是 {'pil'，'cv2'} 中的值。如果未设置此选项，将从 :ref:`paddle.vision.get_image_backend <cn_api_paddle_vision_get_image_backend>` 获得这个值。默认值为   ``None``。

返回
:::::::::

:ref:`cn_api_paddle_io_Dataset`，FashionMNIST 数据集实例。

代码示例
:::::::::

COPY-FROM: paddle.vision.datasets.FashionMNIST
