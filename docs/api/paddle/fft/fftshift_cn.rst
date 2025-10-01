.. _cn_api_paddle_fft_fftshift:

fftshift
-------------------------------

.. py:function:: paddle.fft.fftshift(x, axes=None, name=None)

将零频率项移动到频谱的中心。


参数
:::::::::

- **x** (Tensor) - 输入 Tensor，数据类型为实数或复数。
- **axes** (int，可选) - 进行移动的轴。如果没有指定，默认使用输入 Tensor 中全部的轴。
- **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::

Tensor，形状和数据类型与输入 Tensor 相同，沿着 ``axes`` 移动后的输出。

代码示例
:::::::::

COPY-FROM: paddle.fft.fftshift
