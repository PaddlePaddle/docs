.. _cn_api_paddle_randperm:

randperm
-------------------------------

.. py:function:: paddle.randperm(n, dtype="int64", name=None, *, out=None, device=None, pin_memory=False, requires_grad=False)

返回一个数值在 0 到 n-1、随机排列的 1-D Tensor，数据类型为 ``dtype``。

参数
::::::::::::
  - **n** (int) - 随机序列的上限（不包括在序列中），应该大于 0。
  - **dtype** (str|paddle.dtype|np.dtype，可选) - 输出 Tensor 的数据类型，支持 int32、int64、float32、float64。默认值为 int64。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

关键字参数
::::::::::
  - **out** (Tensor，可选) - 输出 Tensor。
  - **device** (PlaceLike|None，可选) - 返回 Tensor 的期望设备。
  - **pin_memory** (bool，可选) - 若为 True，返回 Tensor 分配在锁页内存中。仅对 CPU Tensor 生效。默认值为 False。
  - **requires_grad** (bool，可选) - 是否由自动微分记录返回 Tensor 上的操作。默认值为 False。

返回
::::::::::
  Tensor：一个数值在 0 到 n-1、随机排列的 1-D Tensor，数据类型为 ``dtype`` 。

代码示例
::::::::::

COPY-FROM: paddle.randperm
