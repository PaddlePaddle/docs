.. _cn_api_fluid_layers_pad2d:

pad2d
-------------------------------

.. py:function::  paddle.fluid.layers.pad2d(input, paddings=[0, 0, 0, 0], mode='constant', pad_value=0.0, data_format='NCHW', name=None)

依照 paddings 和 mode 属性对图像进行2维 ``pad``,如果mode是 ``reflection``，则paddings[0]和paddings[1]必须不大于height-1。宽度维数具有相同的条件。

例如：

.. code-block:: text

  假设X是输入图像:

      X = [[1, 2, 3],
           [4, 5, 6]]

     Case 0:
        paddings = [0, 1, 2, 3],
        mode = 'constant'
        pad_value = 0
        Out = [[0, 0, 1, 2, 3, 0, 0, 0]
               [0, 0, 4, 5, 6, 0, 0, 0]
               [0, 0, 0, 0, 0, 0, 0, 0]]

     Case 1:
        paddings = [0, 1, 2, 1],
        mode = 'reflect'
        Out = [[3, 2, 1, 2, 3, 2]
               [6, 5, 4, 5, 6, 5]
               [3, 2, 1, 2, 3, 2]]

     Case 2:
        paddings = [0, 1, 2, 1],
        mode = 'edge'
        Out = [[1, 1, 1, 2, 3, 3]
               [4, 4, 4, 5, 6, 6]
               [4, 4, 4, 5, 6, 6]]

参数：
  - **input** (Variable) - 具有[N, C, H, W]格式或[N, H, W, C]格式的输入图像。
  - **paddings** (tuple|list|Variable) - 填充区域的大小。如果填充是一个元组，它必须包含四个整数，
    (padding_top, padding_bottom, padding_left, padding_right)。默认:padding =[0,0,0,0]。
  - **mode** (str) - 三种模式:constant(默认)、reflect、edge。默认值:常数
  - **pad_value** (float32) - 以常量模式填充填充区域的值。默认值:0
  - **data_format** (str)  - 可选字符串，选项有: ``NHWC`` , ``NCHW``。指定输入数据的数据格式。默认值:``NCHW``
  - **name** (str|None) - 此层的名称(可选)。如果没有设置，该层将被自动命名。

返回： tensor变量，按照 padding值 和 mode 进行填充

返回类型：variable

**代码示例：**

.. code-block:: python

  import paddle.fluid as fluid
  data = fluid.layers.data(name='data', shape=[3, 32, 32], dtype='float32')
  result = fluid.layers.pad2d(input=data, paddings=[1,2,3,4], mode='reflect')











