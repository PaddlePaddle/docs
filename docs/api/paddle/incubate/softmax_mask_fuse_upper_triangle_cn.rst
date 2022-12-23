.. _cn_api_incubate_softmax_mask_fuse_upper_triangle:

softmax_mask_fuse_upper_triangle
-------------------------------

.. py:function:: paddle.incubate.softmax_mask_fuse_upper_triangle(x)

对输入 ``x`` 进行带 mask 的 softmax 操作，并且总是 mask 住 x 的上三角矩阵部分（不包含对角线部分）。
该 API 主要针对加速 Transformer 架构而设计。将 ``tmp = x + mask, rst = softmax(tmp)`` 两个操作合为一个操作。计算公式为：

.. math::
    out = softmax(LowerTriangular(x))

.. note::
    该 API 只可在 GPU 上运行
参数
:::::::::
    - x (4-D Tensor) - 输入的 Tensor，必须为 4D 的 shape，数据类型为：float16、float32。x 的第四维必须大于等于 32，并且小于 8192。第三维与第四维必须相同。

返回
:::::::::
``Tensor``，维度和数据类型都与 ``x`` 相同，存储运算后的结果


代码示例
::::::::::

COPY-FROM: paddle.incubate.softmax_mask_fuse_upper_triangle
