.. _cn_api_paddle_linalg_vector_norm:

norm
-------------------------------

.. py:function:: paddle.linalg.vector_norm(x, p=2.0, axis=None, keepdim=False, name=None):




将计算给定 Tensor 的向量范数。具体用法请参见 :ref:`norm <_cn_api_paddle_linalg_norm>`。


参数
:::::::::

    - **x** (Tensor) - 输入 Tensor。维度为多维，数据类型为 float32 或 float64。
    - **p** (float，可选) - 范数(ord)的种类。目前支持的值为任何正实数 p 对应的 p 范数。默认值为 2.0 。
    - **axis** (int|list|tuple，可选) - 使用范数计算的轴。如果 ``axis`` 为 None，则忽略 input 的维度，将其当做向量来计算。如果 ``axis`` 为 int 或者 list|tuple，计算 Tensor 对应 axis 上的向量范数。默认值为 `None` 。
    - **keepdim** (bool，可选) - 是否在输出的 Tensor 中保留和输入一样的维度，默认值为 False。当 :attr:`keepdim` 为 False 时，输出的 Tensor 会比输入 :attr:`input` 的维度少一些。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::

 Tensor，在指定 axis 上进行范数计算的结果，与输入 input 数据类型相同。

代码示例
:::::::::

COPY-FROM: paddle.linalg.vector_norm
