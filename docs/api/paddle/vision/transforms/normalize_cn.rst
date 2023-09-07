.. _cn_api_paddle_vision_transforms_normalize:

normalize
-------------------------------

.. py:function:: paddle.vision.transforms.normalize(img, mean, std, data_format='CHW', to_rgb=False)

用均值和标准差归一化输入数据。

参数
:::::::::

    - **img** (PIL.Image|np.array|paddle.Tensor) - 用于归一化的数据。
    - **mean** (list|tuple) - 用于每个通道归一化的均值。
    - **std** (list|tuple) - 用于每个通道归一化的标准差值。
    - **data_format** (str，可选) - 数据的格式，必须为 'HWC' 或 'CHW'。默认值：'CHW'。
    - **to_rgb** (bool，可选) - 是否转换为 ``rgb`` 的格式。默认值：False。

返回
:::::::::

``numpy array`` 或 ``paddle.Tensor``，归一化后的图像。

代码示例
:::::::::

COPY-FROM: paddle.vision.transforms.normalize
