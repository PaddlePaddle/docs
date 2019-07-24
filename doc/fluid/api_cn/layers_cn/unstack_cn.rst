.. _cn_api_fluid_layers_unstack:

unstack
-------------------------------

.. py:function:: paddle.fluid.layers.unstack(x, axis=0, num=None)

实现了unstack层。

沿 ``axis`` 轴，该层对输入 ``x`` 进行unstack运算。

如果 ``axis`` <0，则将其以 :math:`axis+rank(x)` 代之。

如果 ``num`` 为 None，则它可以从 ``x.shape[axis]`` 中推断而来。

如果 ``x.shape[axis]`` <= 0或者Unknown, 则抛出异常 ``ValueError`` 。

参数:
  - **x** (Variable|list(Variable)|tuple(Variable)) – 输入变量
  - **axis** (int|None) – 对输入进行unstack运算所在的轴
  - **num** (int|None) - 输出变量的数目

返回: 经unstack运算后的变量

返回类型: list(Variable)

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    x = fluid.layers.data(name='x', shape=[5, 10], dtype='float32')
    y = fluid.layers.unstack(x, axis=1)







