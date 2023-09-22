.. _cn_api_paddle_vision_transforms_BrightnessTransform:

BrightnessTransform
-------------------------------

.. py:class:: paddle.vision.transforms.BrightnessTransform(value, keys=None)

调整图像的亮度。

参数
:::::::::

    - **value** (float) - 亮度调整范围大小，会从给定参数后的均匀分布[max(0，1 - brightness), 1 + brightness]中随机选择进行实际调整，可以是任何非负数。参数等于 0 时输出原始图像。
    - **keys** (list[str]|tuple[str]，可选) - 与 ``BaseTransform`` 定义一致。默认值：None。

形状
:::::::::

    - img (PIL.Image|np.ndarray|Paddle.Tensor) - 输入的图像数据，数据格式为'HWC'。
    - output (PIL.Image|np.ndarray|Paddle.Tensor) - 返回调整亮度后的图像数据。

返回
:::::::::

    计算 ``BrightnessTransform`` 的可调用对象。

代码示例
:::::::::

COPY-FROM: paddle.vision.transforms.BrightnessTransform
