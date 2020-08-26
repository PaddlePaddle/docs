.. _cn_api_fluid_layers_less_equal:

less_equal
-------------------------------

.. py:function:: paddle.fluid.layers.less_equal(x, y, cond=None, name=None)

:alias_main: paddle.less_equal
:alias: paddle.less_equal,paddle.tensor.less_equal,paddle.tensor.logic.less_equal
:old_api: paddle.fluid.layers.less_equal



该OP逐元素地返回 :math:`x <= y` 的逻辑值，使用重载算子 `<=` 可以有相同的计算函数效果。

参数：
    - **x** (Variable) – 进行比较的第一个输入，是一个多维的Tensor，数据类型可以是float32，float64，int32，int64。 
    - **y** (Variable) – 进行比较的第二个输入，是一个多维的Tensor，数据类型可以是float32，float64，int32，int64。
    - **cond** (Variable，可选) – 如果为None，则创建一个Tensor来作为进行比较的输出结果，该Tensor的shape和数据类型和输入x一致；如果不为None，则将Tensor作为该OP的输出，数据类型和数据shape需要和输入x一致。默认值为None。 
    - **name** （str，可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：输出结果的Tensor，数据的shape和输入x一致。

返回类型：Variable，数据类型为bool类型。

**代码示例**:

.. code-block:: python

     import paddle.fluid as fluid
     import paddle.fluid.layers as layers
     import numpy as np
     label = layers.assign(np.array([1, 3], dtype='int32'))
     limit = layers.assign(np.array([1, 2], dtype='int32'))
     out = fluid.layers.less_equal(x=label, y=limit) #out=[True, False]
     out1 = label<= limit #out1=[True, False]




