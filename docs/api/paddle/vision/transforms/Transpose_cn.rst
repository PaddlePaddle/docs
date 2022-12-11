.. _cn_api_vision_transforms_Transpose:

Transpose
-------------------------------

.. py:class:: paddle.vision.transforms.Transpose(order=(2, 0, 1), keys=None)

将输入的图像数据更改为目标格式。例如，大多数数据预处理是使用 HWC 格式的图片，而神经网络可能使用 CHW 模式输入 Tensor。
输出的图片是 numpy.ndarray 的实例。

参数
:::::::::

    - **order** (list|tuple，可选) - 目标的维度顺序。Default: (2, 0, 1)。
    - **keys** (list[str]|tuple[str]，可选) - 与 ``BaseTransform``。默认值：None。

形状
:::::::::

    - **img** (PIL.Image|np.ndarray|Paddle.Tensor) - 输入的图像数据，数据格式为'HWC'。
    - **output** (np.ndarray|Paddle.Tensor) - 返回更改格式后的数组或 Tensor。如果输入是``PIL.Image``，输出将会自动转换为``np.ndarray``。

返回
:::::::::

    计算 ``Transpose`` 的可调用对象。

代码示例
:::::::::

COPY-FROM: paddle.vision.transforms.Transpose
