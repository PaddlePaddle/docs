.. _cn_api_vision_datasets_Flowers:

Flowers
-------------------------------

.. py:class:: paddle.vision.datasets.Flowers(data_file=None, label_file=None, setid_file=None, mode='train', transform=None, download=True, backend=None)


`Flowers102 <https://www.robots.ox.ac.uk/~vgg/data/flowers/>`_ 数据集的实现。

参数
:::::::::

  - **data_file** (str，可选) - 数据文件路径，如果 ``download`` 参数设置为 ``True``， ``data_file`` 参数可以设置为 ``None``。默认值为 ``None``，默认存放在：``~/.cache/paddle/dataset/flowers``。
  - **label_file** (str，可选) - 标签文件路径，如果 ``download`` 参数设置为 ``True``， ``label_file`` 参数可以设置为 ``None``。默认值为 ``None``，默认存放在：``~/.cache/paddle/dataset/flowers``。
  - **setid_file** (str，可选) - 子数据集下标划分文件路径，如果 ``download`` 参数设置为 ``True`` ， ``setid_file`` 参数可以设置为 ``None``。默认值为 ``None``，默认存放在：``~/.cache/paddle/dataset/flowers``。
  - **mode** (str，可选) - ``'train'`` 或 ``'test'`` 模式两者之一，默认值为 ``'train'``。
  - **transform** (Callable，可选) - 图片数据的预处理，若为 ``None`` 即为不做预处理。默认值为 ``None``。
  - **download** (bool，可选) - 当 ``data_file`` 是 ``None`` 时，该参数决定是否自动下载数据集文件。默认值为 ``True``。
  - **backend** (str，可选) - 指定要返回的图像类型：PIL.Image 或 numpy.ndarray。必须是 {'pil'，'cv2'} 中的值。如果未设置此选项，将从 :ref:`paddle.vision.get_image_backend <cn_api_vision_image_get_image_backend>` 获得这个值。默认值为 ``None``。

返回
:::::::::

:ref:`cn_api_io_cn_Dataset`，Flowers 数据集实例。

代码示例
:::::::::

COPY-FROM: paddle.vision.datasets.Flowers
