.. _cn_api_paddle_rand_like:

rand_like
-------------------------------

.. py:function:: paddle.rand_like(input, name=None, *, dtype=None, device=None, requires_grad=False)

返回一个与输入张量尺寸相同的张量，其元素为从区间 [0, 1) 上均匀分布中采样的随机数。

参数
::::::::::
    - **input** (Tensor) – 输入的多维 Tensor，用于指定输出张量的形状。数据类型可以是 float16、bfloat16、float32、float64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
    - **dtype** (str|np.dtype|paddle.dtype，可选) - 输出 Tensor 的数据类型，支持 float16、bfloat16、float32、float64。当该参数值为 None 时，输出 Tensor 的数据类型与输入 Tensor 的数据类型一致。该参数为仅关键字参数，默认值为 None。
    - **device** (str|paddle.Place，可选) - 指定创建 Tensor 的设备位置。如果为 None，则使用与输入 Tensor 相同的设备。该参数为仅关键字参数，默认值为 None。
    - **requires_grad** (bool，可选) - 是否为创建的 Tensor 计算梯度。该参数为仅关键字参数，默认值为 False。

返回
::::::::::
    Tensor：与输入张量 ``input`` 形状相同的 Tensor，其元素为从区间 [0, 1) 上均匀分布中采样的随机数，数据类型为 ``dtype``。

代码示例
:::::::::::

COPY-FROM: paddle.rand_like
