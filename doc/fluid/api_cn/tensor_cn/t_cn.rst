.. _cn_api_paddle_tensor_t:

t
-------------------------------

.. py:function:: paddle.tensor.t(input, name=None)

该OP对小于等于2维的Tensor进行数据转置。0维和1维Tensor返回本身，2维Tensor等价于perm设置为0，1的 :ref:`cn_api_fluid_layers_transpose` 函数。

参数：
    - **input** (Variable) - 输入：N维(N<=2)Tensor，可选的数据类型为float16, float32, float64, int32, int64。
    - **name** (str, 可选)- 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None

返回： N维Tensor

返回类型：Variable

**示例**:

.. code-block:: python

        # 例1 (0-D tensor)
        x = tensor([0.79])
        paddle.t(x) = tensor([0.79])

        # 例2 (1-D tensor)
        x = tensor([0.79, 0.84, 0.32])
        paddle.t(x) = tensor([0.79, 0.84, 0.32])

        # 例3 (2-D tensor)
        x = tensor([0.79, 0.84, 0.32],
                    [0.64, 0.14, 0.57])
        paddle.t(x) = tensor([0.79, 0.64],
                            [0.84, 0.14],
                            [0.32, 0.57])


**代码示例**:

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    x = fluid.data(name='x', shape=[2, 3], dtype='float32')
    x_transposed = paddle.t(x) # paddle.t 等价于 paddle.tensor.t
    print(x_transposed.shape)
    #(3L, 2L)

