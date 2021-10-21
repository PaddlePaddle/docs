.. _cn_overview_fft:

paddle.fft
---------------------

paddle.fft 目录下包含飞桨框架支持的快速傅里叶变换的相关API。具体如下：

-  :ref:`标准快速傅里叶变换 <standard_ffts>`
-  :ref:`实数傅里叶变换 <real_ffts>`
-  :ref:`厄米特傅里叶变换 <hermitian_ffts>`
-  :ref:`标准快速傅里叶变换 <helper_functions>`

.. _standard_ffts:
标准快速傅里叶变换
==========================

.. csv-table::
    :header: "API名称", "API功能"
    
    " :ref:`paddle.fft.fft <cn_api_fft_fft>` ", "一维离散傅里叶变换"
    " :ref:`paddle.fft.ifft <cn_api_fft_ifft>` ", "一维逆向离散傅里叶变换"
    " :ref:`paddle.fft.fft2 <cn_api_fft_fft2>` ", "二维离散傅里叶变换"
    " :ref:`paddle.fft.ifft2 <cn_api_fft_ifft2>` ", "二维逆向离散傅里叶变换"
    " :ref:`paddle.fft.fftn <cn_api_fft_fftn>` ", "N 维离散傅里叶变换"
    " :ref:`paddle.fft.ifftn <cn_api_fft_ifftn>` ", "N 维逆向离散傅里叶变换"

.. _real_ffts:
实数傅里叶变换
==========================

.. csv-table::
    :header: "API名称", "API功能"
    
    " :ref:`paddle.fft.rfft <cn_api_fft_rfft>` ", "一维离散实数傅里叶变换"
    " :ref:`paddle.fft.irfft <cn_api_fft_irfft>` ", "一维离散实数傅里叶变换的逆变换"
    " :ref:`paddle.fft.rfft2 <cn_api_fft_rfft2>` ", "二维离散实数傅里叶变换"
    " :ref:`paddle.fft.irfft2 <cn_api_fft_irfft2>` ", "二维离散实数傅里叶变换的逆变换"
    " :ref:`paddle.fft.rfftn <cn_api_fft_rfftn>` ", "N 维离散实数傅里叶变换"
    " :ref:`paddle.fft.irfftn <cn_api_fft_irfftn>` ", "N 维离散实数傅里叶变换的逆变换"

.. _hermitian_ffts:
厄米特傅里叶变换
==========================

.. csv-table::
    :header: "API名称", "API功能"
    
    " :ref:`paddle.fft.hfft <cn_api_fft_hfft>` ", "一维离散厄米特傅里叶变换"
    " :ref:`paddle.fft.ihfft <cn_api_fft_ihfft>` ", "一维离散厄米特傅里叶变换的逆变换"
    " :ref:`paddle.fft.hfft2 <cn_api_fft_hfft2>` ", "二维离散厄米特傅里叶变换"
    " :ref:`paddle.fft.ihfft2 <cn_api_fft_ihfft2>` ", "二维离散厄米特傅里叶变换的逆变换"
    " :ref:`paddle.fft.hfftn <cn_api_fft_hfftn>` ", "N 维离散厄米特傅里叶变换"
    " :ref:`paddle.fft.ihfftn <cn_api_fft_ihfftn>` ", "N 维离散厄米特傅里叶变换的逆变换"

.. _helper_functions:
辅助函数
==========================

.. csv-table::
    :header: "API名称", "API功能"
    
    " :ref:`paddle.fft.fftfreq <cn_api_fft_fftfreq>` ", "计算傅里叶变换采样频率"
    " :ref:`paddle.fft.rfftfreq <cn_api_fft_rfftfreq>` ", "计算傅里叶变换采样频率，用于 ``rfft``, ``irfft``"
    " :ref:`paddle.fft.fftshift <cn_api_fft_fftshift>` ", "移动零频率项至频谱中心"
    " :ref:`paddle.fft.ifftshift <cn_api_fft_ifftshift>` ", "fftshift 的逆变换"

背景
==========================
傅里叶分析是将信号表示为一系列周期性成分，并且从这些周期性成分中还原信号的方法。当信号和傅里叶
变换都被替换成离散化的，这个过程称为离散傅里叶变换 (Discrete Fourier Transform, DFT). 
因为快速傅里叶变换算法的高效性，傅里叶变换称为数值计算的一个重要支柱。

离散傅里叶变换将离散的输入表示为离散频率的周期性成分之和，在数字信号处理上有广泛的应用，比如滤
波。在数字信号处理的语境中，离散傅里叶变换的输入一般是定义在时域上的，称为信号(signal)，其输出
定义在频域上的，称为频谱(spectrum).


实现细节
==========================

paddle.fft 的离散傅里叶变换中，一维离散傅里叶变换定义如下：

.. math::

    X_{k} = \sigma \sum_{j=0}^{n-1} x_{j} \exp (\delta i 2 \pi \frac{jk}{n})

n 为傅里叶变换点数，亦即傅里叶变换轴的长度。

..math:: \delta 和变换的方向有关，正向变换中，取值为 -1， 逆向变换中，取值为 1.

..math:: \sigma 为缩放系数，和变换的方向以及缩放方案有关。paddle.fft 中缩放方案有三种：
"forward"，"backward"，"ortho" 之一，默认值为 "backward"。三种缩放模式对应的行为如下：

- "backward": 正向和逆向变换的缩放系数分别为 ``1`` 和 ``1/n``;
- "forward": 正向和逆向变换的缩放系数分别为 ``1/n`` 和 ``1``;
- "ortho": 正向和逆向变换的缩放系数均为 ``1/sqrt(n)``;

对于多维离散傅里叶变换，其定义如下：

.. math::

    X_{k_{1}, k_{2}, \cdots, k_{d}} = \sigma \sum_{j_{d}=0}^{n_{d}-1} \cdots \sum_{j_{2}=0}^{n_{2}-1} \sum_{j_{d}=0}^{n_{1}-1} x_{j_{1}, j_{2}, \cdots ,j_{d}} \exp (\delta i 2 \pi \sum_{l=1}^{d} \frac{j_{l}k_{l}}{n_{l}})


d 是傅里叶变换维数。

..math:: \delta 和变换的方向有关，正向变换中，取值为 -1， 逆向变换中，取值为 1.

..math:: \sigma 为缩放系数，和变换的方向以及缩放方案有关。paddle.fft 中缩放方案有三种：
"forward"，"backward"，"ortho" 之一，默认值为 "backward"。三种缩放模式对应的行为如下：

- "backward": 正向和逆向变换的缩放系数分别为 ``1`` 和 ``1/n``;
- "forward": 正向和逆向变换的缩放系数分别为 ``1/n`` 和 ``1``;
- "ortho": 正向和逆向变换的缩放系数均为 ``1/sqrt(n)``;

其中 

.. math::

    n = \pi_{i=1}^{d} n_{i}





