.. _cn_api_tensor_searchsorted:

searchsorted
-------------------------------

.. py:function:: paddle.searchsorted(sorted_sequence, values, out_int32=False, right=False, name=None)

将根据给定的``values``在``sorted_sequence``的最后一个维度查找合适的索引。

参数：
    - **sorted_sequence** (Tensor) - 输入的多维Tensor，支持的数据类型：float32、float64、int32、int64。该Tensor的数值在其最后一个维度递增。
    - **values** (Tensor) - 输入的多维Tensor，支持的数据类型：float32、float64、int32、int64。
    - **out_int32** (bool，可选) - 输出的数据类型支持int32、int64。默认值为False，表示默认的输出数据类型为int64。
    - **right** (bool，可选) - 根据给定``values``在``sorted_sequence``查找对应的上边界或下边界。默认值为False，表示在``sorted_sequence``的查找给定``values``的下边界。
    - **name** (str，可选) – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：Tensor（与 ``values`` 维度信息一致），如果参数``out_int32``为False，则返回数据类型为int32的Tensor，否则将返回int64的tensor。




**代码示例**：

.. code-block:: python

    import paddle
    
    sorted_sequence = paddle.to_tensor([[1, 3, 5, 7, 9, 11],
                                        [2, 4, 6, 8, 10, 12]], dtype='int32')
    values = paddle.to_tensor([[3, 6, 9, 10], [3, 6, 9, 10]], dtype='int32')
    out1 = paddle.searchsorted(sorted_sequence, values)
    print(out1)
    # Tensor(shape=[2, 4], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
    #        [[1, 3, 4, 5],
    #         [1, 2, 4, 4]])
    out2 = paddle.searchsorted(sorted_sequence, values, right=True)
    print(out2)
    # Tensor(shape=[2, 4], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
    #        [[2, 3, 5, 5],
    #         [1, 3, 4, 5]])
    sorted_sequence_1d = paddle.to_tensor([1, 3, 5, 7, 9, 11, 13])
    out3 = paddle.searchsorted(sorted_sequence_1d, values)     
    print(out3)
    # Tensor(shape=[2, 4], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
    #        [[1, 3, 4, 5],
    #         [1, 3, 4, 5]])
    