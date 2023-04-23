.. _cn_api_paddle_fft_ifft2:

ifft2
-------------------------------

.. py:function:: paddle.fft.ifft2(x, s=None, axes=(-2, -1), norm="backward", name=None)

二维傅里叶变换(``fft2``)的逆变换。在一定的误差范围内，``ifft2(fft2(x)) == x``。


参数
:::::::::

- **x** (Tensor) - 输入 Tensor，数据类型为实数或复数。
- **s** (Sequence[int]，可选) - 输出 Tensor 在每一个傅里叶变换轴上的长度（类似一维逆向傅
  里叶变换中的参数 ``n``）。对于每一个傅里叶变换的轴，如果 ``s`` 中该轴的长度比输入 Tensor
  中对应轴的长度小，输入 Tensor 会被截断。如果 ``s`` 中该轴的长度比输入 Tensor 中对应轴
  的长度大，则输入会被补零。如果 ``s`` 没有指定，则使用输入 Tensor 中由 ``axes`` 指定的各
  个轴的长度。
- **axes** (Sequence[int]，可选) - 傅里叶变换的轴。如果没有指定，默认使用最后两维。
- **norm** (str，可选) - 傅里叶变换的缩放模式，缩放系数由变换的方向和缩放模式同时决定。取值
  必须是 "forward"，"backward"，"ortho" 之一，默认值为 "backward"。三种缩放模式对
  应的行为如下：

  - "backward"：正向和逆向变换的缩放系数分别为 ``1`` 和 ``1/n``；
  - "forward"：正向和逆向变换的缩放系数分别为 ``1/n`` 和 ``1``；
  - "ortho"：正向和逆向变换的缩放系数均为 ``1/sqrt(n)``；

  其中 ``n`` 为 ``s`` 中每个元素连乘。
- **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
:::::::::
Tensor，形状和输入 Tensor 相同，数据类型为复数。由输入 Tensor（可能被截断或者补零之后）在
指定维度进行傅里叶变换的输出。二维傅里叶变换是 N 维傅里叶变换的特例，参考 ``ifftn``。

代码示例
:::::::::

COPY-FROM: paddle.fft.ifft2
