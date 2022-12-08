.. _cn_api_vision_transforms_ContrastTransform:

ContrastTransform
-------------------------------

.. py:class:: paddle.vision.transforms.ContrastTransform(value, keys=None)

调整图像对比度。

参数
:::::::::

    - **value** (float) - 对比度调整范围大小，会从给定参数后的均匀分布[max(0，1 - contrast), 1 + contrast]中随机选择进行实际调整，不能是负数。参数值为 0 时返回原图像。
    - **keys** (list[str]|tuple[str]，可选) - 与 ``BaseTransform`` 定义一致。默认值：None。

形状
:::::::::

    - img (PIL.Image|np.ndarray|Paddle.Tensor) - 输入的图像数据，数据格式为'HWC'。
    - output (PIL.Image|np.ndarray|Paddle.Tensor) - 返回调整对比度后的图像数据。

返回
:::::::::

    计算 ``ContrastTransform`` 的可调用对象。

代码示例
:::::::::

COPY-FROM: paddle.vision.transforms.ContrastTransform
