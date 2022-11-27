.. _cn_api_vision_transforms_to_tensor:

to_tensor
-------------------------------

.. py:function:: paddle.vision.transforms.to_tensor(pic, data_format='CHW')

将 ``PIL.Image`` 或 ``numpy.ndarray`` 转换成 ``paddle.Tensor``。

更多细节请参考 :ref:`cn_api_ToTensor` 。

参数
:::::::::

    - **pic** (PIL.Image|numpy.ndarray) - 输入的图像数据。
    - **data_format** (str，可选) - 返回的张量的格式，必须为 'HWC' 或 'CHW'。默认值：'CHW'。

返回
:::::::::

    ``paddle.Tensor``，转换后的数据。

代码示例
:::::::::

COPY-FROM: paddle.vision.transforms.to_tensor
