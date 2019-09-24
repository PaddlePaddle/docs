.. _cn_api_fluid_layers_pad2d:

pad2d
-------------------------------

.. py:function::  paddle.fluid.layers.pad2d(input, paddings=[0, 0, 0, 0], mode='constant', pad_value=0.0, data_format='NCHW', name=None)

该OP依照 paddings 和 mode 属性对input进行2维 ``pad`` 。

参数：
  - **input** (Variable) -类型为float32的4-D Tensor, format为[N, C, H, W]或[N, H, W, C]。
  - **paddings** (Variable | List[int32]) - 填充大小。如果paddings是一个List，它必须包含四个整数[padding_top, padding_bottom, padding_left, padding_right]。
    如果paddings是Variable, 则是类型为int的1-D Tensor, shape是[4]。默认值为[0,0,0,0]。
  - **mode** (str) - padding的三种模式，分别为constant(默认)、reflect、edge。constant为填充常数pad_value，reflect为填充以input边界值为轴的映射，edge为填充input边界值。具体结果可见以下示例。默认值为constant。
  - **pad_value** (float32) - 以constant模式填充区域时填充的值。默认值为0.0。
  - **data_format** (str)  - 指定input的format，可为 ``NCHW`` 和 ``NHWC`` ，默认值为 ``NCHW`` 。
  - **name** (str, 可选) - 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，缺省值为None。
返回： 对input进行2维 ``pad`` 的结果，类型和input一样的4-D Tensor。

返回类型：Variable

示例：

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



**代码示例：**

.. code-block:: python

  import paddle.fluid as fluid
  data = fluid.layers.data(name='data', shape=[3, 32, 32], dtype='float32')
  result = fluid.layers.pad2d(input=data, paddings=[1,2,3,4], mode='reflect')


