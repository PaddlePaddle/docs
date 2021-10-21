.. _cn_api_paddle_signal_istft:

istft
-------------------------------


.. py:function:: paddle.signal.istft(x, n_fft, hop_length=None, win_length=None, window=None, center=True, normalized=False, onesided=True, length=None, return_complex=False, name=None)

逆短时傅里叶变换。

当输入的窗函数满足NOLA条件时，可以通过逆短时傅里叶变换构建原始信号，NOLA条件：

.. math::
    \sum_{t = -\infty}^{\infty}%
        \text{window}^2[n - t \times H]\ \neq \ 0, \ \text{for } all \ 

上式中符号的意义：  

- :math:`t`: 短时傅里叶变换中的第 :math:`t` 帧输入信号；
- :math:`N`: `n_fft` 参数的值；
- :math:`H`: `hop_length` 参数的值。  


`paddle.signal.istft` 的结果理论上是 `paddle.signal.stft` 的原始输入 `x`，但如果频域的结果是经过修改（如滤波），这种情况下恢复的时域信号是无法保证真实存在的。因此，`paddle.signal.istft` 的结果是对于原始信号的最小二乘估计：[Griffin-Lim optimal estimate](https://ieeexplore.ieee.org/document/1164317) 。

参数
:::::::::

- **x** (Tensor) - 输入数据，是维度为2D或者3D的Tensor，数据类型必须为复数（复信号），其形状为 `[..., fft_size, num_frames]`；
- **n_fft** (int) - 离散傅里叶变换的样本点个数；
- **hop_length** (int，可选) - 对输入分帧时，相邻两帧偏移的样本点个数，默认为 `None` (为 `n_fft//4`)；
- **win_length** (int，可选) - 信号窗的长度，默认为 `None` (为 `n_fft`)；
- **window** (int，可选) - 维度为1D长度为 `win_length` 的Tensor，数据类型可为复数。如果 `win_length < n_fft`，该Tensor将被补长至 `n_fft`。默认为 `None` (长度为 `win_length` 幅值为1的矩形窗)；
- **center** (bool，可选) - 选择是否将输入信号进行补长，使得第 :math:`t \times hop\_length` 个样本点在第 `t` 帧的中心，默认为 `True`；
- **normalized** (bool，可选) - 是否将傅里叶变换的结果乘以值为 `1/sqrt(n)` 的缩放系数；
- **onesided** (bool，可选) - 该参数与 `paddle.signal.stft` 中的有区别，此处表示告知接口输入的 `x` 是否为满足共轭对称性的短时傅里叶变换Tensor的一半。若满足上述条件，且设为 `True`，则 `paddle.signal.istft` 将返回一个实信号，默认为 `True`；
- **length** (int，可选) - 指定输出信号的长度，该信号将从逆短时傅里叶变换的结果中截取。默认为 `None` (返回不截取的信号)；
- **return_complex** (bool，可选) - 表示输出的重构信号是否为复信号。如果 `return_complex` 设为 `True`， `onesided` 必须设为 `False`，默认为 `False`；
- **name** (str，可选) - 输出的名字。一般无需设置，默认值为None。该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` 。 

抛出异常
:::::::::

- ``TypeError`` – 如果输入 `x` 的数据类型不是 `complex64` 或 `complex128`
- ``ValueError``  – 如果输入的窗函数不满足NOLA条件
- ``AssertionError``  – 如果输入 `x` 的维度不为2D或者3D
- ``AssertionError``  – 如果 `hop_length` 小于或等于0，或者大于 `win_length`
- ``AssertionError``  – 如果 `win_length` 小于或等于0或者大于 `n_fft`
- ``AssertionError``  – 如果 `onesided` 为 `True`，但 `fft_size` 不等于 `n_fft//2 + 1`
- ``AssertionError``  – 如果 `onesided` 为 `False`，但 `fft_size` 不等于 `n_fft`
- ``AssertionError``  – 如果输入的窗函数的维度不为1D或者其长度不等于 `win_length`
- ``AssertionError``  – 如果 `return_complex` 为 `True`，但 `onesided` 为 `True`
- ``AssertionError``  – 如果 `return_complex` 为 `False`，但窗函数的数据类型为复数

返回
:::::::::
逆短时傅里叶变换的结果，是重构信号的最小二乘估计Tensor，其形状为 `[..., seq_length]`。

代码示例
:::::::::

COPY-FROM: paddle.signal.istft
