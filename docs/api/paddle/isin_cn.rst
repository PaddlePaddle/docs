.. _cn_api_paddle_isin:

isin
-----------------------------

.. py:function:: paddle.isin(x, test_x, assume_unique=False, invert=False, name=None)

检验 ``x`` 中的每一个元素是否在 ``test_x`` 中。

.. note::
    别名支持: 参数名 ``elements`` 可替代 ``x``，参数名 ``test_elements`` 可替代 ``test_x``，如 ``isin(elements=tensor1, test_elements=tensor2)`` 等价于 ``isin(x=tensor1, test_x=tensor2)`` 。

参数
:::::::::
    - **x** (Tensor) - 输入的 tensor，数据类型为：'bfloat16', 'float16', 'float32', 'float64', 'int32', 'int64'。别名： ``elements``。
    - **test_x** (Tensor) - 用于检验的 tensor，数据类型为：'bfloat16', 'float16', 'float32', 'float64', 'int32', 'int64'。别名： ``test_elements``。
    - **assume_unique** (bool，可选) - 如果设置为 True，表示 ``x`` 与 ``test_x`` 的元素均是唯一的，这种情况可以提升计算的速度。默认值为 False。
    - **invert** (bool，可选) - 是否输出反转的结果。如果为 True，表示将结果反转。默认值为 False。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
``Tensor``，输出的 tensor 与输入 ``x`` 形状相同。

代码示例
:::::::::

COPY-FROM: paddle.isin
