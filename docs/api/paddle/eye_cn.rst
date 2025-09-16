.. _cn_api_paddle_eye:

eye
-------------------------------

.. py:function:: paddle.eye(num_rows, num_columns=None, dtype=None, name=None, *, out=None, device=None, requires_grad=False, pin_memory=False)

构建二维 Tensor(主对角线元素为 1，其他元素为 0)。

.. note::
    别名支持: 参数名 ``n`` 可替代 ``num_rows`` 和 ``m`` 可替代 ``num_columns``，如 ``n=1`` 等价于 ``num_rows=1``， ``m=2`` 等价于 ``num_columns=2``。

参数
::::::::::::
  - **num_rows** (int|Tensor) - 生成 2-D Tensor 的行数，数据类型为非负 int32。别名：``n``。
  - **num_columns** (int|Tensor|None，可选) - 生成 2-D Tensor 的列数，数据类型为非负 int32。若为 None，则默认等于 ``num_rows``。别名：``m``。
  - **dtype** (str|paddle.dtype|np.dtype，可选) - 返回 Tensor 的数据类型。支持 int32、int64、float16、float32、float64、complex64、complex128。
    默认值为 None，此时返回 Tensor 的数据类型为 float32。
  - **name** (str，可选) - 操作的名称，具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

关键字参数
::::::::::::
  - **out** (Tensor，可选) - 用于保存输出结果的 Tensor，默认值为 None。
  - **device** (PlaceLike|None，可选) - 指定返回 Tensor 所在的设备。默认值为 None，表示使用当前全局设备（可通过 ``paddle.device.set_device`` 设置）。
    对于 CPU Tensor，设备为 CPU；对于 CUDA Tensor，设备为当前 CUDA 设备。默认值为 None。
  - **requires_grad** (bool，可选) - 是否在返回的 Tensor 上记录 autograd 的操作。默认值为 False。
  - **pin_memory** (bool，可选) - 若设置为 True，则返回的 Tensor 将分配在锁页内存中。仅对 CPU Tensor 生效。默认值为 False。

返回
::::::::::::
 ``shape`` 为 [num_rows, num_columns]的 Tensor。

代码示例
::::::::::::

COPY-FROM: paddle.eye
