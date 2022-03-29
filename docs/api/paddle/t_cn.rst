.. _cn_api_paddle_tensor_t:

t
-------------------------------

.. py:function:: paddle.t(input, name=None)

该OP对小于等于2维的Tensor进行数据转置。0维和1维Tensor返回本身，2维Tensor等价于perm设置为0，1的 :ref:`cn_api_fluid_layers_transpose` 函数。

参数
::::::::
    - **input** (Tensor) - 输入：N维(N<=2)Tensor，可选的数据类型为float16、float32、float64、int32、int64，默认值为None。
    - **name** (str, 可选)- 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None

返回
::::::::

N维Tensor

示例代码
::::::::
.. code-block:: python
            
             import paddle
             
             # Example 1 (0-D tensor)
             x = paddle.to_tensor([0.79])
             paddle.t(x) # [0.79]
             
             # Example 2 (1-D tensor)
             x = paddle.to_tensor([0.79, 0.84, 0.32])
             paddle.t(x) # [0.79000002, 0.83999997, 0.31999999]
             paddle.t(x).shape # [3]
             
             # Example 3 (2-D tensor)
             x = paddle.to_tensor([[0.79, 0.84, 0.32],
                                  [0.64, 0.14, 0.57]])
             x.shape # [2, 3]
             paddle.t(x)
             # [[0.79000002, 0.63999999],
             #  [0.83999997, 0.14000000],
             #  [0.31999999, 0.56999999]]
             paddle.t(x).shape # [3, 2]
