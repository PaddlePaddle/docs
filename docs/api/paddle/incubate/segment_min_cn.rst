.. _cn_api_incubate_segment_min:

segment_min
-------------------------------

.. py:function:: paddle.incubate.segment_min((data, segment_ids, name=None)


分段求最小值函数。

此运算符，将 ``segment_ids`` 中相同索引对应的 ``data`` 的元素，进行求最小值操作。其中 ``segment_ids`` 是一个单调非减序列。
具体而言，该算子计算一个张量 ``out`` ，使得 

.. math::

    out_i = \min_{j \in \{segment\_ids_j == i \} } data_{j}

其中求均值的索引 ``j`` ，是符合 ``segment_ids[j] == i`` 的所有 ``j`` 。


参数
:::::::::
    - **data** (Tensor) - 张量，数据类型为 float32、float64。
    - **segment_ids** (Tensor) - 一维张量，与输入数据`data`的第一维大小相同，表示`data`分段位置，单调非减。合法的数据类型为 int32、int64。
    - **name** (str, 可选): 操作名称（可选，默认为 None）。 更多信息请参考 :ref:`api_guide_Name` 。

返回
:::::::::
    张量，分段求最小值的结果。空的segment_id对应的默认值为0。

代码示例
:::::::::

.. code-block:: python
        
    import paddle
    data = paddle.to_tensor([[1, 2, 3], [3, 2, 1], [4, 5, 6]], dtype='float32')
    segment_ids = paddle.to_tensor([0, 0, 1], dtype='int32')
    out = paddle.incubate.segment_min(data, segment_ids)
    #Outputs: [[1., 2., 1.], [4., 5., 6.]]

