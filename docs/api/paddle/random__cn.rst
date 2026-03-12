.. _cn_api_paddle_random_:

random\_
-------------------------------

.. py:function:: paddle.random_(x, from=0, to=None, *, generator=None)

将张量自身填充为从离散均匀分布[``from``, ``to`` - 1]范围内采样的随机数。若未指定参数，数值范围通常仅受张量数据类型的限制。
但对于浮点类型，若未指定范围，则默认范围为[0, 2^尾数位数]，以确保每个值均可被精确表示。例如，paddle.to_tensor(1, dtype='float64').uniform_() 将在 [0, 2^53] 范围内均匀分布。
random_ 为 Inplace 版本实现，对输入 ``x`` 采用 Inplace 策略。

参数
::::::::::::

    - **x** (Tensor) - 输入多维 Tensor，可选的数据类型为 'int32'、'int64'、'float32'、'float64'、'float16'、'bfloat16'。
    - **from** (int，可选) - 生成随机值的范围下限，默认为 0。。
    - **to** (int|None，可选) - 生成随机值的范围上限（开区间），默认为 None。
    - **generator** (None) - 随机数生成器的占位参数,当前未实现，保留未来使用。

返回
::::::::::::

Tensor：数值服从范围[``from``, ``to`` - 1]内均匀分布的随机 Tensor。
