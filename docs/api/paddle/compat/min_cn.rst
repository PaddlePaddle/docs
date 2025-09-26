.. _cn_api_paddle_compat_min:

min
-------------------------------

.. note::

    PyTorch 兼容的 :ref:`cn_api_paddle_min` 版本，本接口根据输入参数的不同，包含三种不同的功能。

    使用前请详细参考：`【返回参数类型不一致】torch.min`_ 以确定是否使用此模块。

.. _【返回参数类型不一致】torch.min: https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/model_convert/convert_from_pytorch/api_difference/torch/torch.min.html


.. caution::

    下面列举的三种功能参数输入方式 **互斥**，混用非公共的参数输入方法将会导致报错，请谨慎使用。


=====


.. py:function:: paddle.compat.min(x, out=None)


对整个 input tensor 求最小值，见 `paddle.amin` :ref:`cn_api_paddle_amin`

.. note::

    对输入有多个最小值的情况下， ``paddle.compat.min``  的梯度表现与  ``paddle.amin``  一致：会将梯度平均传回到最小值对应的位置。 ``out``  返回方法与静态图联合使用是被禁止的行为，静态图下将报错。

参数
:::::::::
   - **x** （Tensor）- Tensor，支持数据类型为 bfloat16、float16、float32、float64、int32、int64（CUDA GPU），而 uint8, int32, int64, float32, float64 在 CPU 设备上被支持。
   - **out** (Tensor，可选) - 用于引用式传入输出值，注意：动态图下 out 可以是任意 Tensor，默认值为 None。

返回
:::::::::
   Tensor，最小值运算的 Tensor（0D），数据类型和输入数据类型一致。


=====


.. py:function:: paddle.compat.min(x, dim=None, keepdim=False, out=None)


.. note::

    对输入有多个最小值的情况下，`paddle.compat.min` 将只返回梯度值给  ``indices``  中选择的位置。 ``out``  返回方法与静态图联合使用是被禁止的行为，静态图下将报错。

参数
:::::::::
   - **x** （Tensor）- Tensor，支持数据类型为 bfloat16、float16、float32、float64、int32、int64（CUDA GPU），而 uint8, int32, int64, float32, float64 在 CPU 设备上被支持。
   - **dim** （list|tuple|int，可选）- 求最小值运算的维度。必须在 :math:`[−x.ndim, x.ndim]` 范围内。如果 :math:`dim[i] < 0`，则维度将变为 :math:`x.ndim+dim[i]`，默认值为 None。注意，手动传入  ``dim=None``  是不允许的， ``dim``  不可显式被指定为 None。如果需要对整个 tensor 求解全局最小值，请使用第一种参数写法。
   - **keepdim** (bool，可选) - 是否在输出 Tensor 中保留输入的维度。除非 keepdim 为 True，否则输出 Tensor 的维度将比输入 Tensor 小一维，默认值为 False。注意，不传入  ``dim``  时传入本参数是不允许的！
   - **out** (tuple(Tensor, Tensor)，可选) - 用于引用式传入输出值。 ``values``  在前， ``indices``  在后。注意：动态图下 out 可以是任意 Tensor，默认值为 None。

返回
:::::::::
MinMaxRetType(Tensor, Tensor)，此处的  ``MinMaxRetType``  是一个具名元组，含有  ``values``  （在前）和  ``indices``  （在后）两个域，用法与 tuple 一致。


=====


.. py:function:: paddle.compat.min(x, other, out=None)


与  ``other``  Tensor 计算逐元素最小值。见 `paddle.minimum` :ref:`cn_api_paddle_minimum`

.. note::

     ``out``  返回方法与静态图联合使用是被禁止的行为，静态图下将报错。

参数
:::::::::
   - **x** （Tensor）- Tensor，支持数据类型为 bfloat16、float16、float32、float64、int32、int64（CUDA GPU），而 uint8, int32, int64, float32, float64 在 CPU 设备上被支持。
   - **other** （Tensor）- Tensor，支持类型见  ``x`` 。注意， ``other``  的 shape 必须可以被广播到  ``x``  的 shape。
   - **out** (Tensor，可选) - 用于引用式传入输出值，注意：动态图下 out 可以是任意 Tensor，默认值为 None。

返回
:::::::::
   Tensor，逐元素最小的结果，形状、数据类型与 place 与  ``x``  一致。


=====


代码示例
::::::::::

.. note::

    以下示例为上述三种不同输入方法对应功能的示例。

COPY-FROM: paddle.compat.min
