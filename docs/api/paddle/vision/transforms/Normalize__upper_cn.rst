.. _cn_api_paddle_vision_transforms_Normalize__upper:

Normalize
-------------------------------

.. py:class:: paddle.vision.transforms.Normalize(mean=0.0, std=1.0, data_format='CHW', to_rgb=False, keys=None)

用均值和标准差归一化输入数据。给定 n 个通道的均值(M1,...,Mn)和方差(S1,..,Sn)，Normalize 会在每个通道归一化输入数据。output[channel] = (input[channel] - mean[channel]) / std[channel]

参数
:::::::::

    - **mean** (int|float|list|tuple，可选) - 用于每个通道归一化的均值。
    - **std** (int|float|list|tuple，可选) - 用于每个通道归一化的标准差值。
    - **data_format** (str，可选) - 数据的格式，必须为 'HWC' 或 'CHW'。 默认值为 'CHW'。
    - **to_rgb** (bool，可选) - 是否转换为 ``rgb`` 的格式。默认值为 False。
    - **keys** (list[str]|tuple[str]，可选) - 与 ``BaseTransform`` 定义一致。默认值为 None。

形状
:::::::::

    - img (PIL.Image|np.ndarray|paddle.Tensor) - 输入的图像数据，数据格式为'HWC'。
    - output (PIL.Image|np.ndarray|Paddle.Tensor) - 返回归一化后的图像数据。

返回
:::::::::

计算 ``Normalize`` 的可调用对象。

代码示例
:::::::::

COPY-FROM: paddle.vision.transforms.Normalize
