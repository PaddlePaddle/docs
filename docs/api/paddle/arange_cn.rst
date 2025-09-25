.. _cn_api_paddle_arange:

arange
-------------------------------

.. py:function:: paddle.arange(start=0, end=None, step=1, dtype=None, *, out=None, device=None, requires_grad=False, pin_memory=False, name=None)

返回以步长  ``step``  均匀分隔给定数值区间[  ``start``  ,  ``end``  )的 1-D Tensor，数据类型为  ``dtype``  。

当  ``dtype``  表示浮点类型时，为了避免浮点计算误差，建议给  ``end``  加上一个极小值 epsilon，使边界可以更加明确。

参数
::::::::::

  - **start** (float|int|Tensor) - 区间起点（且区间包括此值）。当  ``start``  类型是 Tensor 时，是形状为[]且数据类型为 int32、int64、float32、float64 的 0-D Tensor。如果仅指定  ``start`` ，而  ``end``  为 None，则区间为[0,  ``start`` )。默认值为 0。
  - **end** (float|int|Tensor，可选) - 区间终点（且通常区间不包括此值）。当  ``end``  类型是 Tensor 时，是形状为[]且数据类型为 int32、int64、float32、float64 的 0-D Tensor。默认值为 None。
  - **step** (float|int|Tensor，可选) - 均匀分割的步长。当  ``step``  类型是 Tensor 时，是形状为[]且数据类型为 int32、int64、float32、float64 的 0-D Tensor。默认值为 1。
  - **dtype** (str|paddle.dtype|np.dtype，可选) - 输出 Tensor 的数据类型，支持 int32、int64、float32、float64。当该参数值为 None 时，输出 Tensor 的数据类型为 int64。默认值为 None。

关键字参数
::::::::::::
  - **out** (Tensor，可选) - 用于存储结果的 Tensor。若指定，将直接写入该 Tensor，默认值为 None。
  - **device** (PlaceLike|None，可选) - 期望创建 Tensor 所在的设备。默认值为 None，表示使用当前全局设备（可通过  ``paddle.device.set_device``  设置）。
  - **requires_grad** (bool，可选) - 是否需要为返回的 Tensor 记录梯度信息。默认值为 False。
  - **pin_memory** (bool，可选) - 若为 True，返回的 CPU Tensor 将分配在锁页内存中。仅对 CPU Tensor 生效。默认值为 False。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::
  Tensor，以步长  ``step``  均匀分割给定数值区间[start, end)后得到的 1-D Tensor，数据类型为  ``dtype``  。


代码示例
::::::::::

COPY-FROM: paddle.arange
