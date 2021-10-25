.. _cn_api_paddle_signal_stft:

stft
-------------------------------


.. py:function:: paddle.signal.stft(x, n_fft, hop_length=None, win_length=None, window=None, center=True, pad_mode='reflect', normalized=False, onesided=True, name=None)

短时傅里叶变换。

短时傅里叶变换将输入的信号先进行分帧，然后逐帧进行离散傅的里叶变换计算，计算的公式如下：

.. math::

    X_t[f] = \sum_{n = 0}^{N-1}
                  \mathrm{window}[n]\ x[t \times H + n]\ 
                  \exp(-{2 \pi j f n}/{N})

上式中符号的意义：  

- :math:`t`: 第 :math:`t` 帧输入信号；
- :math:`f`: 傅里叶变换频域的自变量，如果 ``onesided=False`` , :math:`f` 
  取值范围是 :math:`0 \leq f < n\_fft` ,
  如果 `onesided=True`，取值范围是 
  :math:`0 \leq f < \lfloor n\_fft / 2 \rfloor + 1`； 
- :math:`N`: ``n_fft`` 参数的值；
- :math:`H`: ``hop_length`` 参数的值。  


参数
:::::::::

- **x** (Tensor) - 输入数据，是维度为1D或者2D的Tensor，数据类型可为复数（复信号），其形状
  为 ``[..., seq_length]``；
- **n_fft** (int) - 离散傅里叶变换的样本点个数；
- **hop_length** (int，可选) - 对输入分帧时，相邻两帧偏移的样本点个数，默认为 ``None`` 
  (为 ``n_fft//4``)；
- **win_length** (int，可选) - 信号窗的长度，默认为 ``None`` (为 ``n_fft``)；
- **window** (int，可选) - 维度为1D长度为 ``win_length`` 的Tensor，数据类型可为复数。
  如果 ``win_length < n_fft``，该Tensor将被补长至 ``n_fft``。默认为 ``None`` (长度
  为 ``win_length`` 幅值为1的矩形窗)；
- **center** (bool，可选) - 选择是否将输入信号进行补长，使得第 
  :math:`t \times hop\_length` 个样本点在第 ``t`` 帧的中心，默认为 ``True``；
- **pad_mode** (str，可选) - 当 ``center`` 为 ``True`` 时，确定padding的模式，模式
  的选项可以参考 ``paddle.nn.functional.pad``，默认为 "reflect"；
- **normalized** (bool，可选) - 是否将傅里叶变换的结果乘以值为 ``1/sqrt(n)`` 的缩放系
  数；
- **onesided** (bool，可选) - 当输入为实信号时，选择是否只返回傅里叶变换结果的一半的频点
  值（输入信号和窗函数均为实数时，傅里叶变换结果具有共轭对称性）。如果输入的信号或者窗函数的
  数据类型是复数，则此时不能设置为 ``True``。默认为 ``True``；
- **name** (str，可选) - 输出的名字。一般无需设置，默认值为None。该参数供开发人员打印调试
  信息时使用，具体用法请参见 :ref:`api_guide_Name` 。 

抛出异常
:::::::::

- ``TypeError``  – 如果输入 ``x`` 的数据类型不是 ``float16``, ``float32``, 
  ``float64``, ``complex64`` 或 ``complex128``.
- ``AssertionError``  – 如果输入 ``x`` 的维度不为1D或者2D.
- ``AssertionError``  – 如果 ``hop_length`` 小于或等于0.
- ``AssertionError``  – 如果 ``n_fft`` 小于或等于0或者大于输入信号长度。
- ``AssertionError``  – 如果 ``win_length`` 小于或等于0或者大于 ``n_fft``.
- ``AssertionError``  – 如果输入的窗函数的维度不为1D或者其长度不等于 ``win_length``.
- ``AssertionError``  – 如果输入复信号或复窗函数，但 ``onesided`` 为 ``True``.

返回
:::::::::
短时傅里叶变换的结果，复数Tensor。当输入实信号和实窗函数，如果 ``onesided`` 为 ``True``，
其形状为 ``[..., n_fft//2 + 1, num_frames]``；否则为 ``[..., n_fft, num_frames]``。

代码示例
:::::::::

COPY-FROM: paddle.signal.stft
