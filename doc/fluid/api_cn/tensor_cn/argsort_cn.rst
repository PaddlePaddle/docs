.. _cn_api_tensor_cn_argsort:

argsort
-------------------------------

.. py:function:: paddle.argsort(x, axis=-1, descending=False, name=None)

:alias_main: paddle.argsort
:alias: paddle.argsort,paddle.tensor.argsort,paddle.tensor.search.argsort

对输入变量沿给定轴进行排序，输出排序好的数据的相应索引，其维度和输入相同。默认升序排列，如果需要降序排列设置 ``descending=True`` 。


参数：
    - **x** (Tensor) - 输入的多维 ``Tensor`` ，支持的数据类型：float32、float64、int16、int32、int64、uint8。
    - **axis** (int，可选) - 指定对输入Tensor进行运算的轴， ``axis`` 的有效范围是[-R, R)，R是输入 ``x`` 的Rank， ``axis`` 为负时与 ``axis`` +R 等价。默认值为0。
    - **descending** (bool，可选) - 指定算法排序的方向。如果设置为True，算法按照降序排序。如果设置为False或者不设置，按照升序排序。默认值为False。
    - **name** (str，可选) – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：Tensor, 排序后索引信息（与 ``x`` 维度信息一致），数据类型为int64。


**代码示例**：

.. code-block:: python

    import paddle
    import paddle.imperative as imperative 
    import numpy as np
  
    paddle.enable_imperative()
    input_array = np.array([[[5,8,9,5],
                  [0,0,1,7],
                  [6,9,2,4]],
                  [[5,2,4,2],
                  [4,7,7,9],
                  [1,7,0,6]]]).astype(np.float32)
    x = imperative.to_variable(input_array)
    out1 = paddle.argsort(x=x, axis=-1)
    out2 = paddle.argsort(x=x, axis=0)
    out3 = paddle.argsort(x=x, axis=1)
    print(out1.numpy())
    #[[[0 3 1 2]
    #  [0 1 2 3]
    #  [2 3 0 1]]
    # [[1 3 2 0]
    #  [0 1 2 3]
    #  [2 0 3 1]]]
    print(out2.numpy())
    #[[[0 1 1 1]
    #  [0 0 0 0]
    #  [1 1 1 0]]
    # [[1 0 0 0]
    #  [1 1 1 1]
    #  [0 0 0 1]]]
    print(out3.numpy())
    #[[[1 1 1 2]
    #  [0 0 2 0]
    #  [2 2 0 1]]
    # [[2 0 2 0]
    #  [1 1 0 2]
    #  [0 2 1 1]]]
