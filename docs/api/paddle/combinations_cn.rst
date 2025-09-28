.. _cn_api_paddle_combinations:

combinations
-------------------------------

.. py:function:: paddle.combinations(x, r=2, with_replacement=False, name=None)

对输入 Tensor 计算长度为 r 的情况下的所有组合，当 `with_replacement` 设为 False，可类比 python 内置 API `itertools.combinations` 。
当 `with_replacement` 设为 True，可类比 python 内置 API `itertools.combinations_with_replacement(with_replacement=Ture)`。

参数
::::::::::
    - **x** (Tensor) - 输入 1-D Tensor ，它的数据类型可以是 float16，float32，float64，int32，int64。
    - **r** (int，可选) - 组合的数长度，默认值为 2。
    - **with_replacement** (bool，可选) - 是否允许组合数中出现重复值，默认不允许，默认值为 False。
    - **name** (str，可选) - 具体用法请参见  :ref:`api_guide_Name` ，一般无需设置，默认值为 None。

返回
::::::::::
    - ``out`` (Tensor)：由组合数拼接而成的 Tensor ，和输入 x 类型相同。

代码示例
::::::::::

COPY-FROM: paddle.combinations
