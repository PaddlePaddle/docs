.. _cn_api_fluid_dygraph_PRelu:

PRelu
-------------------------------

.. py:class:: paddle.fluid.dygraph.PRelu(name_scope, mode, param_attr=None)

等式：

.. math::
    y = max(0, x) + \alpha min(0, x)


参数：
          - **name_scope** （string）- 该类的名称。
          - **mode** (string) - 权重共享模式。共提供三种激活方式：

             .. code-block:: text

                all: 所有元素使用同一个权值
                channel: 在同一个通道中的元素使用同一个权值
                element: 每一个元素有一个独立的权值
          - **param_attr** (ParamAttr|None) - 可学习权重 :math:`[\alpha]` 的参数属性。


返回： 输出Tensor与输入tensor的shape相同。

返回类型：  变量（Variable）

**代码示例：**

.. code-block:: python

          import paddle.fluid as fluid
          import numpy as np

          inp_np = np.ones([5, 200, 100, 100]).astype('float32')
          with fluid.dygraph.guard():
              mode = 'channel'
              prelu = fluid.PRelu(
                 'prelu',
                 mode=mode,
                 param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(1.0)))
              dy_rlt = prelu(fluid.dygraph.base.to_variable(inp_np))






