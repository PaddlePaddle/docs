.. _cn_api_paddle_vision_datasets_DatasetFolder:

DatasetFolder
-------------------------------

.. py:class:: paddle.vision.datasets.DatasetFolder(root, loader=None, extensions=None, transform=None, is_valid_file=None)


一种通用的数据加载方式，数据需要以如下的格式存放：

.. code-block:: text

    root/class_a/1.ext
    root/class_a/2.ext
    root/class_a/3.ext

    root/class_b/123.ext
    root/class_b/456.ext
    root/class_b/789.ext


参数
::::::::::::

  - **root** (str) - 根目录路径。
  - **loader** (Callable，可选) - 可以加载数据路径的一个函数，如果该值没有设定，默认使用 ``cv2.imread``。默认值为 None。
  - **extensions** (list[str]|tuple[str]，可选) - 允许的数据后缀列表，``extensions`` 和 ``is_valid_file`` 不可以同时设置。如果该值没有设定，默认为 ``('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')``。默认值为 None。
  - **transform** (Callable，可选) - 图片数据的预处理，若为 ``None`` 即为不做预处理。默认值为 ``None``。
  - **is_valid_file** (Callable，可选) - 根据每条数据的路径来判断是否合法的一个函数。``extensions`` 和 ``is_valid_file`` 不可以同时设置。默认值为 None。

返回
:::::::::

:ref:`cn_api_paddle_io_Dataset`，DatasetFolder 实例。

属性
::::::::::::

  - **classes** (list[str]) - 包含全部类名的列表。
  - **class_to_idx** (dict[str, int]) - 类名到类别索引号的映射字典。
  - **samples** (list[tuple[str, int]]) - 一个列表，其中每项为 ``(样本路径, 类别索引号)`` 形式的元组。
  - **targets** (list[int]) - 数据集中各个图片的类别索引号列表。

代码示例
::::::::::::

COPY-FROM: paddle.vision.datasets.DatasetFolder
