.. _cn_api_linalg_norm:

norm
-------------------------------

.. py:function:: paddle.linalg.norm(x, p='fro', axis=None, keepdim=False, name=None):




将计算给定 Tensor 的矩阵范数（Frobenius 范数）和向量范数（向量 1 范数、2 范数、或者通常的 p 范数）。

.. note::

    此 API 与 ``numpy.linalg.norm`` 存在差异。此 API 支持高阶 Tensor（rank>=3）作为输入，输入 ``axis`` 对应的轴就可以计算出 norm 的值。但是 ``numpy.linalg.norm`` 仅支持一维向量和二维矩阵作为输入。特别需要注意的是，此 API 的 P 阶矩阵范数，实际上将矩阵摊平成向量计算。实际计算的是向量范数，而不是真正的矩阵范数。

参数
:::::::::

    - **x** (Tensor) - 输入 Tensor。维度为多维，数据类型为 float32 或 float64。
    - **p** (float|string，可选) - 范数(ord)的种类。目前支持的值为 `fro`、`inf`、`-inf`、`0`、`1`、`2`，和任何正实数 p 对应的 p 范数。默认值为 `fro` 。
    - **axis** (int|list|tuple，可选) - 使用范数计算的轴。如果 ``axis`` 为 None，则忽略 input 的维度，将其当做向量来计算。如果 ``axis`` 为 int 或者只有一个元素的 list|tuple，``norm`` API 会计算输入 Tensor 的向量范数。如果 axis 为包含两个元素的 list，API 会计算输入 Tensor 的矩阵范数。当 ``axis < 0`` 时，实际的计算维度为 rank(input) + axis。默认值为 `None` 。
    - **keepdim** (bool，可选) - 是否在输出的 Tensor 中保留和输入一样的维度，默认值为 False。当 :attr:`keepdim` 为 False 时，输出的 Tensor 会比输入 :attr:`input` 的维度少一些。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::

 Tensor，在指定 axis 上进行范数计算的结果，与输入 input 数据类型相同。

代码示例
:::::::::

COPY-FROM: paddle.linalg.norm
