.. _cn_api_fluid_layers_affine_channel:

affine_channel
-------------------------------

.. py:function:: paddle.fluid.layers.affine_channel(x, scale=None, bias=None, data_layout='NCHW', name=None,act=None)

对输入的每个 channel 应用单独的仿射变换。用于将空间批处理范数替换为其等价的固定变换。

输入也可以是二维张量，并在二维应用仿射变换。

参数：
  - **x** (Variable):特征图输入可以是一个具有NCHW阶或NHWC阶的4D张量。它也可以是二维张量和应用于第二维度的仿射变换。
  - **scale** (Variable): 形状为(C)的一维输入，第C个元素为输入的第C通道仿射变换的尺度因子。
  - **bias** (Variable):形状为(C)的一维输入，第C个元素是输入的第C个通道的仿射变换的偏置。
  - **data_layout** (string, default NCHW): NCHW 或 NHWC，如果输入是一个2D张量，可以忽略该参数
  - **name** (str, default None): 此层的名称
  - **act** (str, default None): 应用于该层输出的激活函数

返回： out (Variable): 与x具有相同形状和数据布局的张量。

**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid
    data = fluid.layers.data(name='data', shape=[3, 32, 32],
                             dtype='float32')
    input_scale = fluid.layers.create_parameter(shape=[3],
                             dtype="float32")
    input_bias = fluid.layers.create_parameter(shape=[3],
                             dtype="float32")
    out = fluid.layers.affine_channel(data,scale=input_scale,
                             bias=input_bias)





