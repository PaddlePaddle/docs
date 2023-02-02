.. _cn_api_paddle_complex:

complex
-------------------------------

.. py:function:: paddle.complex(real, imag, name=None)


给定实部和虚部，返回一个复数 Tensor。


参数
:::::::::
    - **real** (Tensor) - 实部，数据类型为：float32 或 float64。
    - **imag** (Tensor) - 虚部，数据类型和 ``real`` 相同。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
输出 Tensor，数据类型是 complex64 或者 complex128，与 ``real`` 和 ``imag`` 的数值精度一致。

.. note::
    ``paddle.complex`` 遵守 broadcasting，如您想了解更多，请参见 `Tensor 介绍`_ .

    .. _Tensor 介绍: ../../guides/beginner/tensor_cn.html#id7

代码示例
:::::::::

COPY-FROM: paddle.complex
