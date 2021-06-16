.. _cn_api_tensor_tile: 

tile
-------------------------------

.. py:function:: paddle.tile(x, repeat_times, name=None)

根据参数 ``repeat_times`` 对输入 ``x`` 的各维度进行复制。 平铺后，输出的第 ``i``  个维度的值等于 ``x.shape[i]*repeat_times[i]`` 。

``x`` 的维数和 ``repeat_times`` 中的元素数量应小于等于6。

参数
:::::::::
    - x (Tensor) - 输入的Tensor，数据类型为：bool、float32、float64、int32或int64。
    - repeat_times (list|tuple|Tensor) - 指定输入 ``x`` 每个维度的复制次数。如果 ``repeat_times`` 的类型是list或tuple，它的元素可以是整数或者数据类型为int32的1-D Tensor。如果 ``repeat_times`` 的类型是Tensor，则是数据类型为int32的1-D Tensor。
    - name (str，可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name` 。

返回
:::::::::
``Tensor`` ，数据类型与 ``x`` 相同。返回值的第i维的大小等于 ``x[i] * repeat_times[i]`` 。

代码示例
:::::::::

.. code-block:: python

    import paddle
    import numpy as np
    
    np_data = np.array([1, 2, 3]).astype('int32')
    data = paddle.to_tensor(np_data)
    out = paddle.tile(data, repeat_times=[2, 1])
    np_out = out.numpy()
    # [[1, 2, 3], [1, 2, 3]]
    
    out = paddle.tile(data, repeat_times=[2, 2])
    np_out = out.numpy()
    # [[1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3]]
    
    np_repeat_times = np.array([2, 1]).astype("int32")
    repeat_times = paddle.to_tensor(np_repeat_times)
    out = paddle.tile(data, repeat_times=repeat_times)
    np_out = out.numpy()
    # [[1, 2, 3], [1, 2, 3]]
