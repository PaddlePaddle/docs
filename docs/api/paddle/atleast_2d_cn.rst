.. _cn_api_paddle_atleast_2d:

atleast_2d
==========

.. py:function:: paddle.atleast_2d(*inputs, name=None)

将输入转换为张量并返回至少为 ``2`` 维的视图。 ``2`` 维或更高维的输入会被保留。

图例说明
=========
以下图例展示了 `paddle.atleast_2d` 函数对不同维度输入的影响：

.. figure:: /images/api_legend/paddle.atleast_2d.png
   :align: center
   :alt: paddle.atleast_2d

   图示：输入不同维度的张量，经过 `paddle.atleast_2d` 函数的升维效果。

1. **输入为 0 维张量（标量）**：

   - 输入：`5` （标量）
   - 输出：`[[5]]` （扩展为二维张量）

2. **输入为 1 维张量（向量）**：

   - 输入：`[1, 2, 3]` （1 维张量）
   - 输出：`[[1, 2, 3]]` （扩展为二维张量）

3. **输入为 2 维张量（矩阵）**：

   - 输入：`[[1, 2], [3, 4]]` （2 维张量）
   - 输出：`[[1, 2], [3, 4]]` （无需改变）

参数
====
- **inputs** (Tensor | list(Tensor)) - 一个或多个 Tensor，数据类型为： ``float16``, ``float32``, ``float64``, ``int16``, ``int32``, ``int64``, ``int8``, ``uint8``, ``complex64``, ``complex128``, ``bfloat16`` 或 ``bool``。
- **name** (str, 可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回值
======
Tensor 或由 Tensor 组成的 list。当只有一个输入时，返回一个 Tensor。当有多个输入时，返回由 Tensor 组成的 list。

代码示例
========
COPY-FROM: paddle.atleast_2d
