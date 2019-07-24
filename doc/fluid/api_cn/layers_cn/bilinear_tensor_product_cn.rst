.. _cn_api_fluid_layers_bilinear_tensor_product:

bilinear_tensor_product
-------------------------------

.. py:function:: paddle.fluid.layers.bilinear_tensor_product(x, y, size, act=None, name=None, param_attr=None, bias_attr=None)

该层对两个输入执行双线性张量积。

例如:

.. math::
       out_{i} = x * W_{i} * {y^\mathrm{T}}, i=0,1,...,size-1

在这个公式中：
  - :math:`x`: 第一个输入，包含M个元素，形状为[batch_size, M]
  - :math:`y`: 第二个输入，包含N个元素，形状为[batch_size, N]
  - :math:`W_{i}`: 第i个被学习的权重，形状是[M, N]
  - :math:`out_{i}`: out的第i个元素，形状是[batch_size, size]
  - :math:`y^\mathrm{T}`: :math:`y_{2}` 的转置

参数：
    - **x** (Variable): 2-D 输入张量，形状为 [batch_size, M]
    - **y** (Variable): 2-D 输入张量，形状为 [batch_size, N]
    - **size** (int): 此层的维度，
    - **act** (str, default None): 应用到该层输出的激活函数
    - **name** (str, default None): 该层的名称
    - **param_attr** (ParamAttr, default None): 可学习参数/权重（w） 的参数属性
    - **bias_attr** (ParamAttr, default None): 偏差的参数属性，如果设置为False，则不会向输出单元添加偏差。如果设置为零，偏差初始化为零。默认值:None

返回： Variable: 一个形为[batch_size, size]的2-D张量

**代码示例：**

.. code-block:: python

  import paddle.fluid as fluid
  layer1 = fluid.layers.data("t1", shape=[-1, 5], dtype="float32")
  layer2 = fluid.layers.data("t2", shape=[-1, 4], dtype="float32")
  tensor = fluid.layers.bilinear_tensor_product(x=layer1, y=layer2, size=1000)




