.. _cn_api_fluid_layers_argsort:

argsort
-------------------------------

.. py:function:: paddle.fluid.layers.argsort(input,axis=-1,name=None)

对输入变量沿给定轴进行 **升序** 排列，输出排序好的数据和相应的索引，其维度和输入相同。**暂不支持降序排列**。


参数：
    - **input** (Variable) - 输入的多维 ``Tensor`` ，支持的数据类型：float32、float64。
    - **axis** (int)- 指定对输入Tensor进行运算的轴， ``axis`` 的有效范围是[-1, R)，R是输入 ``x`` 的Rank。默认值为-1，表示最后一维。
    - **name** (str，可选) - 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name`，默认值为None。

返回：一组已排序的输出（维度与数据类型和 ``input`` 相同）和索引（数据类型为int64），

返回类型：tuple

**代码示例**：

.. code-block:: python

  import paddle.fluid as fluid
  import numpy as np

  in1 = np.array([[[5,8,9,5],
                   [0,0,1,7],
                   [6,9,2,4]],
                  [[5,2,4,2],
                   [4,7,7,9],
                   [1,7,0,6]]]).astype(np.float32)
  with fluid.dygraph.guard():
      x = fluid.dygraph.to_variable(in1)
      out1 = fluid.layers.argsort(input=x, axis=-1) # same as axis==2
      out2 = fluid.layers.argsort(input=x, axis=0)
      out3 = fluid.layers.argsort(input=x, axis=1)
      print(out1[0].numpy())
      # [[[5. 5. 8. 9.]
      #   [0. 0. 1. 7.]
      #   [2. 4. 6. 9.]]
      #  [[2. 2. 4. 5.]
      #   [4. 7. 7. 9.]
      #   [0. 1. 6. 7.]]]
      print(out1[1].numpy())
      # [[[0 3 1 2]
      #   [0 1 2 3]
      #   [2 3 0 1]]
      #  [[1 3 2 0]
      #   [0 1 2 3]
      #   [2 0 3 1]]]
      print(out2[0].numpy())
      # [[[5. 2. 4. 2.]
      #   [0. 0. 1. 7.]
      #   [1. 7. 0. 4.]]
      #  [[5. 8. 9. 5.]
      #   [4. 7. 7. 9.]
      #   [6. 9. 2. 6.]]]
      print(out3[0].numpy())
      # [[[0. 0. 1. 4.]
      #   [5. 8. 2. 5.]
      #   [6. 9. 9. 7.]]
      #  [[1. 2. 0. 2.]
      #   [4. 7. 4. 6.]
      #   [5. 7. 7. 9.]]]
