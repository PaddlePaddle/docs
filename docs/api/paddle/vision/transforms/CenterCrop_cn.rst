.. _cn_api_paddle_vision_transforms_CenterCrop:

CenterCrop
-------------------------------

.. py:class:: paddle.vision.transforms.CenterCrop(size, keys=None)

对输入图像进行裁剪，保持图片中心点不变。

参数
:::::::::

    - **size** (int|list|tuple) - 输出图像的形状大小。
    - **keys** (list[str]|tuple[str]，可选) - 与 ``BaseTransform`` 定义一致。默认值：None。

形状
:::::::::

    - img (PIL.Image|np.ndarray|Paddle.Tensor) - 输入的图像数据，数据格式为'HWC'。
    - output (PIL.Image|np.ndarray|Paddle.Tensor) - 返回裁剪后的图像数据。

返回
:::::::::

    计算 ``CenterCrop`` 的可调用对象。

代码示例
:::::::::

COPY-FROM: paddle.vision.transforms.CenterCrop
