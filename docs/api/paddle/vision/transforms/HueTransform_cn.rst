.. _cn_api_vision_transforms_HueTransform:

HueTransform
-------------------------------

.. py:class:: paddle.vision.transforms.HueTransform(value, keys=None)

调整图像的色调。

参数
:::::::::

    - **value** (float) - 色调调整范围大小，会从给定参数后的均匀分布[-hue, hue]中随机选择进行实际调整，参数值需要在 0 到 0.5 之间，参数值为 0 时返回原始图像。
    - **keys** (list[str]|tuple[str]，可选) - 与 ``BaseTransform`` 定义一致。默认值：None。

形状
:::::::::

    - img (PIL.Image|np.ndarray|Paddle.Tensor) - 输入的图像数据，数据格式为'HWC'。
    - output (PIL.Image|np.ndarray|Paddle.Tensor) - 返回调整色调后的图像数据。

返回
:::::::::

    计算 ``HueTransform`` 的可调用对象。

代码示例
:::::::::

COPY-FROM: paddle.vision.transforms.HueTransform
