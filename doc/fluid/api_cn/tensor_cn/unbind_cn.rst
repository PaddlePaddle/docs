.. _cn_api_paddle_tensor_unbind
unbind
-------------------------------

.. py:function:: paddle.tensor.unbind(input, axis=0)

:alias_main: paddle.unbind
:alias: paddle.unbind,paddle.tensor.unbind,paddle.tensor.manipulation.unbind



该OP将输入Tensor按照指定的维度分割成多个子Tensor。

**参数**：
       - **input** (Variable) - 输入变量，数据类型为float32，float64，int32，int64的多维Tensor。
       - **axis** (int32|int64，可选) - 数据类型为int32或int64,表示需要分割的维度。如果axis < 0，则划分的维度为rank(input) + axis。默认值为0。

**返回**：分割后的Tensor列表。

**返回类型**：列表(Variable)，数据类型为int32，int64，float32，float64。

**代码示例**：

.. code-block:: python
    
    import paddle
    # input is a variable which shape is [3, 4, 5]
    input = paddle.fluid.data(
        name="input", shape=[3, 4, 5], dtype="float32")
    [x0, x1, x2] = paddle.tensor.unbind(input, axis=0)
    # x0.shape [4, 5]
    # x1.shape [4, 5]
    # x2.shape [4, 5]
    [x0, x1, x2, x3] = paddle.tensor.unbind(input, axis=1)
    # x0.shape [3, 5]
    # x1.shape [3, 5]
    # x2.shape [3, 5]
    # x3.shape [3, 5]
