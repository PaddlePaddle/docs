.. _cn_api_paddle_isin:

isin
-----------------------------

.. py:function:: paddle.isin(x, test_x, assume_unique=False, invert=False, name=None)

检验 ``x`` 中的每一个元素是否在 ``test_x`` 中。

参数
:::::::::
    - **x** (Tensor) - 输入的 tensor，数据类型为：'bfloat16', 'float16', 'float32', 'float64', 'int32', 'int64'。
    - **test_x** (Tensor) - 用于检验的 tensor，数据类型为：'bfloat16', 'float16', 'float32', 'float64', 'int32', 'int64'。
    - **assume_unique** (bool，可选) - 如果设置为 True，表示 ``x`` 与 ``test_x`` 的元素均是唯一的，这种情况可以提升计算的速度。默认值为 False。
    - **invert** (bool，可选) - 是否输出反转的结果。如果为 True，表示将结果反转。默认值为 False。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
``Tensor``，输出的 tensor 与输入 ``x`` 形状相同。

代码示例
:::::::::

COPY-FROM: paddle.isin
