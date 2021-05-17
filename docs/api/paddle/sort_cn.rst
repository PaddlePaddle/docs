.. _cn_api_tensor_sort:

sort
-------------------------------

.. py:function:: paddle.sort(x, axis=-1, descending=False, name=None)



对输入变量沿给定轴进行排序，输出排序好的数据，其维度和输入相同。默认升序排列，如果需要降序排列设置 ``descending=True`` 。


参数：
    - **x** (Tensor) - 输入的多维 ``Tensor`` ，支持的数据类型：float32、float64、int16、int32、int64、uint8。
    - **axis** (int，可选) - 指定对输入Tensor进行运算的轴， ``axis`` 的有效范围是[-R, R)，R是输入 ``x`` 的Rank， ``axis`` 为负时与 ``axis`` +R 等价。默认值为0。
    - **descending** (bool，可选) - 指定算法排序的方向。如果设置为True，算法按照降序排序。如果设置为False或者不设置，按照升序排序。默认值为False。
    - **name** (str，可选) – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：Tensor, 排序后的输出（与 ``x`` 维度相同、数据类型相同）。


**代码示例**：

.. code-block:: python

    import paddle

    x = paddle.to_tensor([[[5,8,9,5],
                        [0,0,1,7],
                        [6,9,2,4]],
                        [[5,2,4,2],
                        [4,7,7,9],
                        [1,7,0,6]]], dtype='float32')
    out1 = paddle.sort(x=x, axis=-1)
    out2 = paddle.sort(x=x, axis=0)
    out3 = paddle.sort(x=x, axis=1)
    print(out1)
    #[[[5. 5. 8. 9.]
    #  [0. 0. 1. 7.]
    #  [2. 4. 6. 9.]]
    # [[2. 2. 4. 5.]
    #  [4. 7. 7. 9.]
    #  [0. 1. 6. 7.]]]
    print(out2)
    #[[[5. 2. 4. 2.]
    #  [0. 0. 1. 7.]
    #  [1. 7. 0. 4.]]
    # [[5. 8. 9. 5.]
    #  [4. 7. 7. 9.]
    #  [6. 9. 2. 6.]]]
    print(out3)
    #[[[0. 0. 1. 4.]
    #  [5. 8. 2. 5.]
    #  [6. 9. 9. 7.]]
    # [[1. 2. 0. 2.]
    #  [4. 7. 4. 6.]
    #  [5. 7. 7. 9.]]]
    