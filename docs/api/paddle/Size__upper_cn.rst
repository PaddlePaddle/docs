.. _cn_api_paddle_Size__upper:

Size
-------------------------------

.. py:class:: paddle.Size(iterable=())
``paddle.Tensor.size()`` 的返回类型，用于描述张量的维度大小。作为 ``tuple`` 的子类，支持所有常见的序列操作（如索引、切片、拼接等）。

**参数**
:::::::::
    - **iterable** (iterable，可选) - 表示维度的整数序列。默认值为 ``()``。

**返回**
:::::::::
    - ``Size``：表示张量维度的特殊元组子类。

**代码示例**
:::::::::

COPY-FROM: paddle.Size
