.. _cn_api_paddle_fft_fftfreq:

fftfreq
-------------------------------

.. py:function:: paddle.fft.fftfreq(n, d=1.0, dtype=None, name=None)

返回离散傅里叶变换的频率窗口(frequency bins)中心序列，以 "循环/采样间隔" 为单位。例如，采
样间隔以秒为单位，则频率的单位是 "循环/秒"。

对于窗口长度 n 和采样间隔 d，输出的频率序列 f 排布如下：

    f = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / (d*n)   （当 n 为偶数）
    f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)   （当 n 为奇数）

参数
:::::::::

- **n** (int) - 窗长度（傅里叶变换点数）。
- **d** (float，可选) - 采样间隔，采样率的倒数，默认值为 1。
- **dtype** (str，可选) - 返回 Tensor 的数据类型，默认为
  ``paddle.get_default_dtype()`` 返回的类型。
- **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
:::::::::

Tensor，形状为 ``[n]``，数据类型为 ``dtype`` 指定的数据类型，包含频率窗口中心序列。

代码示例
:::::::::

COPY-FROM: paddle.fft.fftfreq
