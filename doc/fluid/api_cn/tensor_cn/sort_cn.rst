.. _cn_api_tensor_sort:

sort
-------------------------------

.. py:function:: paddle.sort(input, axis=-1, descending=False, out=None, name=None)

对输入变量沿给定轴进行排序，输出排序好的数据和相应的索引，其维度和输入相同。**默认升序排列，如果需要降序排列设置** ``descending=True`` 。


参数：
    - **input** (Variable) - 输入的多维 ``Tensor`` ，支持的数据类型：float32、float64、int16、int32、int64、uint8。
    - **axis** (int，可选) - 指定对输入Tensor进行运算的轴， ``axis`` 的有效范围是[-R, R)，R是输入 ``x`` 的Rank， ``axis`` 为负时与 ``axis`` +R 等价。默认值为0。
    - **descending** (bool，可选) - 指定算法排序的方向。如果设置为True，算法按照降序排序。如果设置为False或者不设置，按照升序排序。默认值为False。
    - **out** (Variable, 可选) – 指定存储运算结果的Tensor（与 ``input`` 维度相同、数据类型相同）。如果设置为None或者不设置，将创建新的Tensor存储运算结果，默认值为None。
    - **name** (str，可选) – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：一组已排序的输出（与 ``input`` 维度相同、数据类型相同）和索引（数据类型为int64）。

返回类型：tuple[Variable]

**代码示例**：

.. code-block:: python

  import paddle
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
      out1 = paddle.sort(input=x, axis=-1) # same as axis==2
      out2 = paddle.sort(input=x, axis=0)
      out3 = paddle.sort(input=x, axis=1)
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

