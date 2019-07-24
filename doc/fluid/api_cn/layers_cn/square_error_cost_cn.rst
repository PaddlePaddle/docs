.. _cn_api_fluid_layers_square_error_cost:

square_error_cost
-------------------------------

.. py:function:: paddle.fluid.layers.square_error_cost(input,label)

方差估计层（Square error cost layer）

该层接受输入预测值和目标值，并返回方差估计

对于预测值X和目标值Y，公式为：

.. math::

    Out = (X-Y)^{2}

在以上等式中：
    - **X** : 输入预测值，张量（Tensor)
    - **Y** : 输入目标值，张量（Tensor）
    - **Out** : 输出值，维度和X的相同

参数：
    - **input** (Variable) - 输入张量（Tensor），带有预测值
    - **label** (Variable) - 标签张量（Tensor），带有目标值

返回：张量变量，存储输入张量和标签张量的方差

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    y = fluid.layers.data(name='y', shape=[1], dtype='float32')
    y_predict = fluid.layers.data(name='y_predict', shape=[1], dtype='float32')
    cost = fluid.layers.square_error_cost(input=y_predict, label=y)









