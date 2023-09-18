.. _cn_api_paddle_t:

t
-------------------------------

.. py:function:: paddle.t(input, name=None)

对小于等于 2 维的 Tensor 进行数据转置。0 维和 1 维 Tensor 返回本身，2 维 Tensor 等价于 perm 设置为 0，1 的 :ref:`cn_api_paddle_transpose` 函数。

参数
::::::::
    - **input** (Tensor) - 输入：N 维(N<=2)Tensor，可选的数据类型为 float16、float32、float64、int32、int64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::
Tensor，0 维和 1 维 Tensor 返回本身，2 维 Tensor 返回转置 Tensor。

代码示例
:::::::::

.. code-block:: text

        # 例 1 (0-D tensor)
        x = tensor([0.79])
        paddle.t(x) = tensor([0.79])

        # 例 2 (1-D tensor)
        x = tensor([0.79, 0.84, 0.32])
        paddle.t(x) = tensor([0.79, 0.84, 0.32])

        # 例 3 (2-D tensor)
        x = tensor([0.79, 0.84, 0.32],
                    [0.64, 0.14, 0.57])
        paddle.t(x) = tensor([0.79, 0.64],
                            [0.84, 0.14],
                            [0.32, 0.57])


代码示例
::::::::::::

.. code-block:: python

    import paddle
    x = paddle.ones(shape=[2, 3], dtype='int32')
    x_transposed = paddle.t(x)
    print(x_transposed.shape)
    # [3, 2]
