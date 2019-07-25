.. _cn_api_fluid_backward_gradients:

gradients
-------------------------------

.. py:function:: paddle.fluid.backward.gradients(targets, inputs, target_gradients=None, no_grad_set=None)

将目标梯度反向传播到输入。

参数：  
  - **targets** (Variable|list[Variable]) – 目标变量
  - **inputs** (Variable|list[Variable]) – 输入变量
  - **target_gradients** (Variable|list[Variable]|None) – 目标的梯度变量，应与目标变量形状相同；如果设置为None，则以1初始化所有梯度变量
  - **no_grad_sethread** (set[string]) – 在Block 0中不具有梯度的变量，所有block中被设置 ``stop_gradient=True`` 的变量将被自动加入该set


返回：数组，包含与输入对应的梯度。如果一个输入不影响目标函数，则对应的梯度变量为None

返回类型：(list[Variable])

**示例代码**

.. code-block:: python

            import paddle.fluid as fluid

            x = fluid.layers.data(name='x', shape=[2,8,8], dtype='float32')
            x.stop_gradient=False
            y = fluid.layers.conv2d(x, 4, 1, bias_attr=False)
            y = fluid.layers.relu(y)
            y = fluid.layers.conv2d(y, 4, 1, bias_attr=False)
            y = fluid.layers.relu(y)
            z = fluid.gradients([y], x)
            print(z)