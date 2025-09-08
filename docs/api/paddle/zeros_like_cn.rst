.. _cn_api_paddle_zeros_like:

zeros_like
-------------------------------

.. py:function:: paddle.zeros_like(x, dtype=None, name=None, *, device=None, requires_grad=False, pin_memory=False)


返回一个和 ``x`` 具有相同的形状的全零 Tensor，数据类型为 ``dtype`` 或者和 ``x`` 相同。

参数
::::::::::
    - **x** (Tensor) – 输入的多维 Tensor，数据类型可以是 bool，float16, float32，float64，int32，int64。输出 Tensor 的形状和 ``x`` 相同。如果 ``dtype`` 为 None，则输出 Tensor 的数据类型与 ``x`` 相同。
    - **dtype** (str|paddle.dtype|np.dtype，可选) - 输出 Tensor 的数据类型，支持 bool，float16, float32，float64，int32，int64。当该参数值为 None 时，输出 Tensor 的数据类型与 ``x`` 相同。默认值为 None。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

关键字参数
::::::::::::
    - **device** (PlaceLike|None，可选) - 期望创建 Tensor 所在的设备。若为 None，则与 ``x`` 保持一致。
    - **requires_grad** (bool，可选) - 是否需要为返回的 Tensor 记录梯度信息。默认值为 False。
    - **pin_memory** (bool，可选) - 若为 True，返回的 CPU Tensor 将分配在锁页内存中。仅对 CPU Tensor 生效。默认值为 False。

返回
::::::::::
    Tensor：和 ``x`` 具有相同的形状全零 Tensor，数据类型为 ``dtype`` 或者和 ``x`` 相同。


代码示例
::::::::::

COPY-FROM: paddle.zeros_like
