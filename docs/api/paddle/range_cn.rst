.. _cn_api_paddle_range:

range
-------------------------------

.. py:function:: paddle.range(start=0, end=None, step=1, dtype=None, *, out=None, device=None, requires_grad=False, name=None)

返回一个形状为 [$\lfloor \dfrac{end-start}{step} \rfloor + 1$] 的 1-D Tensor(对应 **闭区间** [ ``start`` , ``end`` ])，数据类型为 ``dtype`` 。

参数
::::::::::

  - **start** (float|int|Tensor) - 区间起点（且区间包括此值）。当 ``start`` 类型是 Tensor 时，是形状为[]且数据类型为 int32、int64、float32、float64 的 0-D Tensor。如果仅指定 ``start``，而 ``end`` 为 None，则区间为[0, ``start``)。默认值为 0。
  - **end** (float|int|Tensor，可选) - 区间终点（且通常区间不包括此值）。当 ``end`` 类型是 Tensor 时，是形状为[]且数据类型为 int32、int64、float32、float64 的 0-D Tensor。默认值为 None。
  - **step** (float|int|Tensor，可选) - 均匀分割的步长。当 ``step`` 类型是 Tensor 时，是形状为[]且数据类型为 int32、int64、float32、float64 的 0-D Tensor。默认值为 1。
  - **dtype** (str|np.dtype，可选) - 输出 Tensor 的数据类型，支持 int32、int64、float32、float64。当该参数值为 None 时，输出 Tensor 的数据类型为 int64。默认值为 None。

关键字参数
::::::::::::
  - **out** (Tensor，可选) - 用于存储结果的 Tensor。若指定，将直接写入该 Tensor，默认值为 None。
  - **device** (PlaceLike|None，可选) - 期望创建 Tensor 所在的设备。默认值为 None，表示使用当前全局设备（可通过 ``paddle.device.set_device`` 设置）。
  - **requires_grad** (bool，可选) - 是否需要为返回的 Tensor 记录梯度信息。默认值为 False。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::
  返回一个形状为 [$\lfloor \dfrac{end-start}{step} \rfloor + 1$] 的 1-D Tensor(对应 **闭区间** [ ``start`` , ``end`` ])，数据类型为 ``dtype`` 。


代码示例
::::::::::

COPY-FROM: paddle.range
