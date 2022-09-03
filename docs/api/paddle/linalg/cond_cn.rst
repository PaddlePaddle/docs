.. _cn_api_linalg_cond:

cond
-------------------------------

.. py:function:: paddle.linalg.cond(x, p=None, name=None)


根据范数种类 ``p`` 计算一个或一批矩阵的条件数，也可以通过 paddle.cond 来调用。

参数
::::::::::::

    - **x** (Tensor)：输入可以是形状为 ``(*, m, n)`` 的矩阵 Tensor， ``*`` 为零或更大的批次维度，此时 ``p`` 为 `2` 或 `-2`；也可以是形状为 ``(*, n, n)`` 的可逆（批）方阵 Tensor，此时 ``p`` 为任意已支持的值。数据类型为 float32 或 float64 。
    - **p** (float|string，可选)：范数种类。目前支持的值为 `fro` 、 `nuc` 、 `1` 、 `-1` 、 `2` 、 `-2` 、 `inf` 、 `-inf`。默认值为 `None`，即范数种类为 `2` 。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
Tensor，条件数的计算结果，数据类型和输入 ``x`` 的一致。

代码示例
::::::::::

COPY-FROM: paddle.linalg.cond
