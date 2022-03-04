.. _cn_api_paddle_nn_layer_common_Identity:

Identity
-------------------------------

.. py:class:: paddle.nn.Identity(*args, **kwargs)


**等效层** 。对于输入Tensor :math:`X` ，计算公式为：

.. math::

    Out = X


参数
:::::::::

- **args** - 任意的参数（没有使用）
- **kwargs** – 任意的关键字参数（没有使用）

形状
:::::::::

- 输入：形状为 :math:`[batch\_size, n1, n2, ...]` 的多维Tensor。
- 输出：形状为 :math:`[batch\_size, n1, n2, ...]` 的多维Tensor。

代码示例
:::::::::

.. code-block:: python

    import paddle
    input_tensor = paddle.randn(shape=[3, 2])
    layer = paddle.nn.Identity()
    out = layer(input_tensor)
    # input_tensor: [[-0.32342386 -1.200079  ]
    #                [ 0.7979031  -0.90978354]
    #                [ 0.40597573  1.8095392 ]]
    # out: [[-0.32342386 -1.200079  ]
    #      [ 0.7979031  -0.90978354]
    #      [ 0.40597573  1.8095392 ]]
