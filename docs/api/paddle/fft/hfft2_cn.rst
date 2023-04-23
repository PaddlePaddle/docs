.. _cn_api_paddle_fft_hfft2:

hfft2
-------------------------------

.. py:function:: paddle.fft.hfft2(x, s=None, axes=(-2, -1), norm="backward", name=None)

通过快速傅里叶变换(FFT)算法计算二维厄米特(Hermitian)傅里叶变换。


参数
:::::::::

- **x** (Tensor) - 输入数据，其数据类型为复数类型。
- **s** (Sequence[int]，可选) - 输出 Tensor 在傅里叶变换轴的长度（类似一维傅里叶变换中
  的参数 ``n``）。
- **axes** (Sequence[int]，可选) - 傅里叶变换的轴。如果没有指定，默认使用最后两个轴。
- **norm** (str，可选) - 傅里叶变换的缩放模式，缩放系数由变换的方向和缩放模式同时决定。取
  值必须是 "forward"，"backward"，"ortho" 之一，默认值为 "backward"。三种缩放模式对应
  的行为如下：

  - "backward"：正向和逆向变换的缩放系数分别为 ``1`` 和 ``1/n``；
  - "forward"：正向和逆向变换的缩放系数分别为 ``1/n`` 和 ``1``；
  - "ortho"：正向和逆向变换的缩放系数均为 ``1/sqrt(n)``；

  其中 ``n`` 为 ``s`` 中每个元素连乘。
- **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
:::::::::
Tensor，数据类型为实数。由输入 Tensor（可能被截断或者补零之后）在指定维度进行傅里叶变换的输
出。二维傅里叶变换为 N 维傅里叶(``hfftn``)变换的特例。

代码示例
:::::::::

COPY-FROM: paddle.fft.hfft2
