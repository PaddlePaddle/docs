.. _cn_api_fluid_layers_group_norm:

group_norm
-------------------------------

.. py:function::  paddle.fluid.layers.group_norm(input, groups, epsilon=1e-05, param_attr=None, bias_attr=None, act=None, data_layout='NCHW', name=None)

参考论文： `Group Normalization <https://arxiv.org/abs/1803.08494>`_

参数：
  - **input** (Variable)：输入张量变量
  - **groups** (int)：从 channel 中分离出来的 group 的数目
  - **epsilon** (float)：为防止方差除零，增加一个很小的值
  - **param_attr** (ParamAttr|None)：可学习标度的参数属性 :math:`g`,如果设置为False，则不会向输出单元添加标度。如果设置为0，偏差初始化为1。默认值:None
  - **bias_attr** (ParamAttr|None)：可学习偏置的参数属性 :math:`b ` , 如果设置为False，则不会向输出单元添加偏置量。如果设置为零，偏置初始化为零。默认值:None。
  - **act** (str):将激活应用于输出的 group normalizaiton
  - **data_layout** (string|NCHW): 只支持NCHW。
  - **name** (str):这一层的名称（可选）

返回： Variable: 一个张量变量，它是对输入进行 group normalization 后的结果。

**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid
    data = fluid.layers.data(name='data', shape=[8, 32, 32],
                             dtype='float32')
    x = fluid.layers.group_norm(input=data, groups=4)










