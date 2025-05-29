.. _cn_api_paddle_expand:

expand
-------------------------------

.. py:function:: paddle.expand(x, shape, name=None)

根据 ``shape`` 指定的形状扩展 ``x``，扩展后，``x`` 的形状和 ``shape`` 指定的形状一致。

``x`` 的维数和 ``shape`` 的元素数应小于等于 6，并且 ``shape`` 中的元素数应该大于等于 ``x`` 的维数。扩展的维度的维度值应该为 1。

**示例图解说明**：
    Original Tensor 显示了一个形状为 [3] 的一维张量 [1, 2, 3]。通过 paddle.expand 方法（参数 ``shape = [2, 3]``）被广播扩展为一个形状为 [2, 3] 的二维张量。

    .. figure:: ../../images/api_legend/expand.png
       :width: 500
       :alt: 示例一图示
       :align: center


参数
:::::::::
    - **x** (Tensor) - 输入的 Tensor，数据类型为：bool、float16、float32、float64、int32、int64、uint8 或 uint16。
    - **shape** (tuple|list|Tensor) - 给定输入 ``x`` 扩展后的形状，若 ``shape`` 为 list 或者 tuple，则其中的元素值应该为整数或者是形状为 1-D 或 0-D 的 Tensor，若 ``shape`` 类型为 Tensor，则其应该为 1-D Tensor。值为-1 表示保持相应维度的形状不变。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
``N-D Tensor``，数据类型与 ``x`` 相同。

代码示例
:::::::::

COPY-FROM: paddle.expand
