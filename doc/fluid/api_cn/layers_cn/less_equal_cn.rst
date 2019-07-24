.. _cn_api_fluid_layers_less_equal:

less_equal
-------------------------------

.. py:function:: paddle.fluid.layers.less_equal(x, y, cond=None)

该层逐元素地返回 :math:`x <= y` 的逻辑值，和重载算子 `<=` 相同。

参数：
    - **x** (Variable) - *less_equal* 的第一个操作数
    - **y** (Variable) - *less_equal* 的第二个操作数
    - **cond** (Variable|None) - 可选的输出变量，存储 *less_equal* 的结果

返回：存储 *less_equal* 的输出的张量变量。

返回类型：变量（Variable）

**代码示例**:

.. code-block:: python

     import paddle.fluid as fluid
     out = fluid.layers.less_equal(x=label, y=limit)




