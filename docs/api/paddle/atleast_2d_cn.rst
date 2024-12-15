.. _cn_api_paddle_atleast_2d:

atleast_2d
-------------------------------

.. py:function:: paddle.atleast_2d(*inputs, name=None)

将输入转换为张量并返回至少为 ``2`` 维的视图。 ``2`` 维或更高维的输入会被保留。

图例说明
========

以下图例展示了 `paddle.atleast_2d` 函数对不同维度输入的影响：

.. figure:: /images/api_legend/paddle.atleast_2d.png
   :align: center
   :alt: 显示 paddle.atleast_2d 的升维过程

1. **输入为标量（0 维）**
   - 输入：`5`
   - 输出：`[[5]]`

2. **输入为一维张量（1 维）**
   - 输入：`[1, 2, 3]`
   - 输出：`[[1, 2, 3]]`

3. **输入为二维张量（2 维）**
   - 输入：`[[1, 2], [3, 4]]`
   - 输出：保持不变

参数
::::::::::::

    - **inputs** (Tensor|list(Tensor)) - 一个或多个 Tensor，数据类型为： ``float16``, ``float32``, ``float64``, ``int16``, ``int32``, ``int64``, ``int8``, ``uint8``, ``complex64``, ``complex128``, ``bfloat16`` 或 ``bool``。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
Tensor 或者由 Tensor 组成的 list。当只有一个输入的时候返回一个 Tensor，当有多个输入的时候返回由 Tensor 组成的 list。

代码示例
::::::::::::

COPY-FROM: paddle.atleast_2d
