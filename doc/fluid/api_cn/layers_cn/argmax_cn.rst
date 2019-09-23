.. _cn_api_fluid_layers_argmax:

argmax
-------------------------------

.. py:function:: paddle.fluid.layers.argmax(x, axis=0)

**argmax**

该OP沿 ``axis`` 计算输入 ``x`` 的最大元素的索引。

参数：
    - **x** (Variable) - 输入的多维 ``Tensor`` ，支持的数据类型：float32、float64、int8、int16、int32、int64。
    - **axis** (int，可选) - 指定对输入Tensor进行运算的轴， ``axis`` 的有效范围是[-1, R)，R是输入 ``x`` 的Rank， ``-1`` 表示最后一维。默认值为0。

返回： ``Tensor`` ，数据类型int64

返回类型：Variable

**代码示例**：

.. code-block:: python

  import paddle.fluid as fluid
  import numpy as np

  in1 = np.array([[[5,8,9,5],
               [0,0,1,7],
               [6,9,2,4]],

              [[5,2,4,2],
               [4,7,7,9],
               [1,7,0,6]]])
  with fluid.dygraph.guard():
      x = fluid.dygraph.to_variable(in1)
      out1 = fluid.layers.argmax(x=x, axis=-1)
      out2 = fluid.layers.argmax(x=x, axis=0)
      out3 = fluid.layers.argmax(x=x, axis=1)
      out4 = fluid.layers.argmax(x=x, axis=2)
      print(out1.numpy())
      # [[2 3 1]
      #  [0 3 1]]
      print(out2.numpy())
      # [[0 0 0 0]
      #  [1 1 1 1]
      #  [0 0 0 1]]
      print(out3.numpy())
      # [[2 2 0 1]
      #  [0 1 1 1]]
      print(out4.numpy())
      # [[2 3 1]
      #  [0 3 1]]
