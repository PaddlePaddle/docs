.. _cn_api_paddle_vision_transforms_RandomVerticalFlip:

RandomVerticalFlip
-------------------------------

.. py:class:: paddle.vision.transforms.RandomVerticalFlip(prob=0.5, keys=None)

基于概率来执行图片的垂直翻转。

参数
:::::::::

    - **prob** (float) - 执行图片垂直翻转的概率，默认值为 0.5。
    - **keys** (list[str]|tuple[str]，可选) - 与 ``BaseTransform`` 定义一致。默认值：None。

形状
:::::::::

    - img (PIL.Image|np.ndarray|Paddle.Tensor) - 输入的图像数据，数据格式为'HWC'。
    - output (PIL.Image|np.ndarray|Paddle.Tensor) - 返回概率执行垂直翻转后的图像数据。

返回
:::::::::

    计算 ``RandomVerticalFlip`` 的可调用对象。

代码示例
:::::::::

COPY-FROM: paddle.vision.transforms.RandomVerticalFlip
