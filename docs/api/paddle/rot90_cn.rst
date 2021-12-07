.. _cn_api_tensor_rot90:

rot90
-------------------------------

.. py:function:: paddle.rot90(x, k=1, axes=[0, 1], name=None):



该API沿axes指定的平面将n维tensor旋转90度。当k为正数，旋转方向为从axes[0]到axes[1]，当k为负数，旋转方向为从axes[1]到axes[0]，k的绝对值表示旋转次数。

参数
::::::::::

    - **x** (Tensor) - 输入张量。维度为多维，数据类型为bool, int32, int64, float16, float32或float64。
    - **k** (int, 可选) - 旋转方向和次数，默认值：1。
    - **axes** (list|tuple, 可选) - axes指定旋转的平面，维度必须为2。默认值为[0, 1]。
    - **name** (str|None, 可选) - 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` 。默认值为None。

返回
::::::::::

    - 在指定平面axes上翻转指定次数后的张量，与输入x数据类型相同。


代码示例
::::::::::

.. code-block:: python

    import paddle

    data = paddle.arange(4)
    data = paddle.reshape(data, (2, 2))
    print(data) 
    #[[0, 1],
    # [2, 3]]
    y = paddle.rot90(data, 1, [0, 1])
    print(y)
    #[[1, 3],
    # [0, 2]]
    y= paddle.rot90(data, -1, [0, 1])
    print(y) 
    #[[2, 0],
    # [3, 1]]
    data2 = paddle.arange(8)
    data2 = paddle.reshape(data2, (2,2,2))
    print(data2) 
    #[[[0, 1],
    #  [2, 3]],
    # [[4, 5],
    #  [6, 7]]]
    y = paddle.rot90(data2, 1, [1, 2])
    print(y)   
    #[[[1, 3],
    #  [0, 2]],
    # [[5, 7],
    #  [4, 6]]]
