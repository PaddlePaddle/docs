.. _cn_api_incubate_softmax_mask_fuse_upper_triangle:

softmax_mask_fuse_upper_triangle
-------------------------------

.. py:function:: paddle.incubate.softmax_mask_fuse_upper_triangle(x)

该op是对输入 ``x`` 进行被mask的softmax操作，该op总是mask住x的上三角矩阵部分（不包含对角线部分）。该op主要针对加速Transformer架构而设计。将 ``tmp = x + mask, rst = softmax(tmp)`` 两个操作合为一个操作。计算公式为：

.. math::
    out = softmax(LowerTriangular(x))

.. note::
    该API只可在GPU上运行

参数
:::::::::
    - x (4-D Tensor) - 输入的Tensor，必须为4D的shape，数据类型为：float16、float32。x的第四维必须大于等于32，并且小于8192。第三维与第四维必须相同。

返回
:::::::::
``Tensor``，维度和数据类型都与 ``x`` 相同，存储运算后的结果


代码示例
::::::::::

.. code-block:: python
    
    # required: gpu
    import paddle
    import paddle.incubate as incubate
    x = paddle.rand((1, 1, 32, 32))
    rst = incubate.softmax_mask_fuse_upper_triangle(x)
    # [[[[1.        , 0.        , 0.        , ..., 0., 0., 0.],
    #    [0.45324376, 0.54675621, 0.        , ..., 0., 0., 0.],
    #    [0.32674268, 0.28156221, 0.39169508, ..., 0., 0., 0.]
    #     ... ]]]
