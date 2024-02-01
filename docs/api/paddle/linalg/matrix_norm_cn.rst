.. _cn_api_paddle_linalg_matrix_norm:

matrix_norm
-------------------------------

.. py:function:: paddle.linalg.matrix_norm(x, p='fro', axis=[-2,-1], keepdim=False, name=None):




将计算给定 Tensor 的矩阵范数。具体用法请参见 :ref:`norm <_cn_api_paddle_linalg_norm>`。


参数
:::::::::

    - **x** (Tensor) - 输入 Tensor。维度为多维，数据类型为 float32 或 float64。
    - **p** (int|float|string，可选) - 范数(ord)的种类。目前支持的值为 `fro`、`nuc`、`inf`、`-inf`、`1`、`2`、`-1`、`-2`。默认值为 `fro` 。
    - **axis** (list|tuple，可选) - 使用范数计算的轴。``axis`` 为 [-2,-1],否则 ``axis`` 必须为长度为 2 的 list|tuple。
    - **keepdim** (bool，可选) - 是否在输出的 Tensor 中保留和输入一样的维度，默认值为 False。当 :attr:`keepdim` 为 False 时，输出的 Tensor 会比输入 :attr:`input` 的维度少一些。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::

 Tensor，在指定 axis 上进行范数计算的结果，与输入 input 数据类型相同。

代码示例
:::::::::

COPY-FROM: paddle.linalg.matrix_norm
