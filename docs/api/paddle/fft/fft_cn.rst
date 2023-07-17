.. _cn_api_paddle_fft_fft:

fft
-------------------------------

.. py:function:: paddle.fft.fft(x, n=None, axis=-1, norm="backward", name=None)

一维离散傅里叶变换。

通过快速傅里叶变换（FFT）算法对 M 维 Tensor 中的一维计算离散傅里叶变换。

参数
:::::::::

- **x** (Tensor) - 输入 Tensor，数据类型为实数或复数。
- **n** (int，可选) - 输出 Tensor 中傅里叶变换轴的长度。如果 ``n`` 比输入 Tensor 中对应轴的长度小，输入数据会被截断。如果 ``n`` 比输入 Tensor 中对应轴的长度大，则输入会被补零。如果 ``n`` 没有被指定，则使用输入 Tensor 中由 ``axis`` 指定的轴的长度。
- **axis** (int，可选) - 傅里叶变换的轴。如果没有指定，默认使用最后一维。
- **norm** (str，可选) - 傅里叶变换的缩放模式，缩放系数由变换的方向和缩放模式同时决定。取
  值必须是 "forward"，"backward"，"ortho" 之一，默认值为 "backward"。三种缩放模式对应
  的行为如下：

  - "backward"：正向和逆向变换的缩放系数分别为 ``1`` 和 ``1/n``；
  - "forward"：正向和逆向变换的缩放系数分别为 ``1/n`` 和 ``1``；
  - "ortho"：正向和逆向变换的缩放系数均为 ``1/sqrt(n)``；

- **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
:::::::::

Tensor，形状和输入 Tensor 相同，数据类型为复数。由输入 Tensor（可能被截断或者补零之后）在指定维度进行傅里叶变换的输出。

代码示例
:::::::::

COPY-FROM: paddle.fft.fft
