.. _cn_api_paddle_nn_layer_common_Identity:

Identity
-------------------------------

.. py:class:: paddle.nn.Identity(*args, **kwargs)


**等效层** 。对于每个输入Tensor :math:`X` ，计算公式为：

.. math::

    Out = X


参数
:::::::::

- 任意参数，均没有使用

形状
:::::::::

- 输入：形状为 :math:`[batch\_size, n1, n2, ...]` 的多维Tensor。
- 输出：形状为 :math:`[batch\_size, n1, n2, ...]` 的多维Tensor。

代码示例
:::::::::

.. code-block:: python

    import paddle
    layer = paddle.nn.Identity(54)
    input_tensor = paddle.to_tensor(paddle.randn(shape=[3, 2]))
    input_tensor.stop_gradient=False
    input_tensor = input_tensor+1
    input_tensor.register_hook(lambda grad: print('input grad', grad))
    print('input_tensor.grad', input_tensor.grad)
    out = m(input_tensor)
    # input_tensor: [[-0.32342386 -1.200079  ]
    #                [ 0.7979031  -0.90978354]
    #                [ 0.40597573  1.8095392 ]]
    # out: [[-0.32342386 -1.200079  ]
    #      [ 0.7979031  -0.90978354]
    #      [ 0.40597573  1.8095392 ]]
    out.backward()
    print(out.shape, paddle.sum(input_tensor), paddle.sum(out))
