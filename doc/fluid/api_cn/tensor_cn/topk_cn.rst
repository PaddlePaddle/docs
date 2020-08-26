.. _cn_api_tensor_cn_topk:

topk
-------------------------------

.. py:function:: paddle.topk（x, k, axis=None, largest=True, sorted=True, name=None）

该OP沿着可选的 ``axis`` 查找topk最大或者最小的结果和结果所在的索引信息。
如果是一维Tensor，则直接返回topk查询的结果。如果是多维Tensor，则在指定的轴上查询topk的结果。

参数
:::::::::
    - **x** （Tensor） - 输入的多维 ``Tensor`` ，支持的数据类型：float32、float64、int32、int64。
    - **k** （int，Tensor） - 在指定的轴上进行top寻找的数量。 
    - **axis** （int，可选） - 指定对输入Tensor进行运算的轴， ``axis`` 的有效范围是[-R, R），R是输入 ``x`` 的Rank， ``axis`` 为负时与 ``axis`` + R 等价。默认值为-1。
    - **largest** （bool，可选） - 指定算法排序的方向。如果设置为True，排序算法按照降序的算法排序，否则按照升序排序。默认值为True。
    - **sorted** （bool，可选） - 控制返回的结果是否按照有序返回，默认为True。在gpu上总是返回有序的结果。
    - **name** （str，可选） – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回
:::::::::
tuple（Tensor）, 返回topk的结果和结果的索引信息。结果的数据类型和输入 ``x`` 一致。索引的数据类型是int64。

代码示例
:::::::::


.. code-block:: python

    import numpy as np
    import paddle

    paddle.disable_static()

    data_1 = np.array([1, 4, 5, 7])
    tensor_1 = paddle.to_tensor(data_1)
    value_1, indices_1 = paddle.topk(tensor_1, k=1)
    print(value_1.numpy())
    # [7]
    print(indices_1.numpy())
    # [3] 
    data_2 = np.array([[1, 4, 5, 7], [2, 6, 2, 5]])
    tensor_2 = paddle.to_tensor(data_2)
    value_2, indices_2 = paddle.topk(tensor_2, k=1)
    print(value_2.numpy())
    # [[7]
    #  [6]]
    print(indices_2.numpy())
    # [[3]
    #  [1]]
    value_3, indices_3 = paddle.topk(tensor_2, k=1, axis=-1)
    print(value_3.numpy())
    # [[7]
    #  [6]]
    print(indices_3.numpy())
    # [[3]
    #  [1]]
    value_4, indices_4 = paddle.topk(tensor_2, k=1, axis=0)
    print(value_4.numpy())
    # [[2 6 5 7]]
    print(indices_4.numpy())
    # [[1 1 0 0]]

