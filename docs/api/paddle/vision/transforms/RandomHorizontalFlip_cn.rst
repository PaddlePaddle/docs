.. _cn_api_vision_transforms_RandomHorizontalFlip:

RandomHorizontalFlip
-------------------------------

.. py:class:: paddle.vision.transforms.RandomHorizontalFlip(prob=0.5, keys=None)

基于概率来执行图片的水平翻转。

参数
:::::::::

    - **prob** (float，可选) - 图片执行水平翻转的概率，取值范围为[0, 1]，默认值为 0.5。
    - **keys** (list[str]|tuple[str]，可选) - 与 ``BaseTransform`` 定义一致。默认值：None。

形状
:::::::::

    - img (PIL.Image|np.ndarray|Paddle.Tensor) - 输入的图像数据，数据格式为'HWC'。
    - output (PIL.Image|np.ndarray|Paddle.Tensor) - 返回概率执行水平翻转后的图像数据。

返回
:::::::::

    计算 ``RandomHorizontalFlip`` 的可调用对象。

代码示例
:::::::::

COPY-FROM: paddle.vision.transforms.RandomHorizontalFlip
