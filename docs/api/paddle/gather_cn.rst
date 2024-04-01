.. _cn_api_paddle_gather:

gather
-------------------------------

.. py:function:: paddle.gather(x, index, axis=None, name=None)

根据索引 index 获取输入 ``x`` 的指定 ``axis`` 维度的条目，并将它们拼接在一起。

.. code-block:: text

        Given:

        X = [[1, 2],
             [3, 4],
             [5, 6]]

        Index = [1, 2]

        axis = 0

        Then:

        Out = [[3, 4],
               [5, 6]]

参数
::::::::::::
        - **x** (Tensor) - 输入 Tensor，秩 ``rank >= 1``，支持的数据类型包括 int32、int64、float32、float64、complex64、complex128 和 uint8 (CPU)、float16（GPU） 。
        - **index** (Tensor) - 索引 Tensor，秩 ``rank = 0`` 或者 ``rank = 1``，数据类型为 int32 或 int64。
        - **axis** (Tensor) - 指定 index 获取输入的维度，``axis`` 的类型可以是 int 或者 Tensor，当 ``axis`` 为 Tensor 的时候其数据类型为 int32 或者 int64。默认值为 None，当``axis``为 None 的时候其值为 0。
        - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
Tensor，当 index 为一维 Tensor 时，返回和输入 Tensor 的形状相同的 Tensor。当 index 为零维 Tensor 时，返回 Tensor 相对于输入 Tensor 会降维， axis 指向的维度会被降维。


代码示例
::::::::::::

COPY-FROM: paddle.gather
