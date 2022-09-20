.. _cn_api_paddle_fft_hfft:

hfft
-------------------------------


.. py:function:: paddle.fft.hfft(x, n=None, axis=-1, norm="backward", name=None)

通过快速傅里叶变换(FFT)算法计算一维厄米特(Hermitian)傅里叶变换。


参数
:::::::::

- **x** (Tensor) - 输入数据，其数据类型为复数。
- **n** (int，可选) - 输出 Tensor 在傅里叶变换轴的长度。输入 Tensor 在该轴的长度必须为
  ``n//2+1``，如果输入 Tensor 的长度大于 ``n//2+1``，输入 Tensor 会被截断。如果输入
  Tensor 的长度小于 ``n//2+1``，则输入 Tensor 会被补零。如果 ``n`` 没有被指定，则取
  ``2*(m-1)``，其中，``m`` 是输入 Tensor 在 ``axis`` 维的长度。
- **axis** (int，optional) - 傅里叶变换的轴。如果没有指定，默认是使用最后一维。
- **norm** (str，可选) - 傅里叶变换的缩放模式，缩放系数由变换的方向和缩放模式同时决定。取
  值必须是 "forward", "backward", "ortho" 之一，默认值为 "backward"。三种缩放模式对
  应的行为如下：

  - "backward"：正向和逆向变换的缩放系数分别为 ``1`` 和 ``1/n``；
  - "forward"：正向和逆向变换的缩放系数分别为 ``1/n`` 和 ``1``；
  - "ortho"：正向和逆向变换的缩放系数均为 ``1/sqrt(n)``；

- **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
:::::::::
Tensor，数据类型为实数。由输入 Tensor（可能被截断或者补零之后）在指定维度进行傅里叶变换的输
出。如果指定 n，则输出 Tensor 在傅立叶变换轴的长度为 n，否则为 ``2*(m-1)``，其中``m``
是输入 Tensor 在 ``axis`` 维的长度。

代码示例
:::::::::

COPY-FROM: paddle.fft.hfft
