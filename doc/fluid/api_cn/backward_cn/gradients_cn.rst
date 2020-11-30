.. _cn_api_fluid_backward_gradients:

gradients
-------------------------------


.. py:function:: paddle.fluid.backward.gradients(targets, inputs, target_gradients=None, no_grad_set=None)

:api_attr: 声明式编程模式（静态图)



将目标梯度反向传播到输入。

参数：  
  - **targets** (Variable|list[Variable]) – 目标变量
  - **inputs** (Variable|list[Variable]) – 输入变量
  - **target_gradients** (Variable|list[Variable]，可选) – 目标的梯度变量，应与目标变量形状相同；如果设置为None，则以1初始化所有梯度变量
  - **no_grad_set** (set[Variable|str]，可选) – 在 `block0` ( :ref:`api_guide_Block` ) 中要忽略梯度的 :ref:`api_guide_Variable` 的名字的集合。所有的 :ref:`api_guide_Block` 中带有 ``stop_gradient = True`` 的所有 :ref:`api_guide_Variable` 的名字都会被自动添加到此集合中。如果该参数不为 ``None``，则会将该参数集合的内容添加到默认的集合中。默认值为 ``None``。


返回：数组，包含与输入对应的梯度。如果一个输入不影响目标函数，则对应的梯度变量为None

返回类型：(list[Variable])

**示例代码**

.. code-block:: python

            import paddle.fluid as fluid

            x = fluid.data(name='x', shape=[None,2,8,8], dtype='float32')
            x.stop_gradient=False
            y = fluid.layers.conv2d(x, 4, 1, bias_attr=False)
            y = fluid.layers.relu(y)
            y = fluid.layers.conv2d(y, 4, 1, bias_attr=False)
            y = fluid.layers.relu(y)
            z = fluid.gradients([y], x)
            print(z)