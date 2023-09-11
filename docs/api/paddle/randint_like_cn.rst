.. _cn_api_paddle_randint_like:

randint_like
-------------------------------

.. py:function:: paddle.randint_like(x, low=0, high=None, dtype=None, name=None)

返回服从均匀分布的、范围在[``low``, ``high``)的随机 Tensor，输出的形状与 x 的形状一致，当数据类型 ``dtype`` 为 None 时（默认），输出的数据类型与 x 的数据类型一致，当数据类型 ``dtype`` 不为 None 时，将输出用户指定的数据类型。当 ``high`` 为 None 时（默认），均匀采样的区间为[0, ``low``)。

参数
::::::::::
    - **x** (Tensor) – 输入的多维 Tensor，数据类型可以是 bool，int32，int64，float16，float32，float64。输出 Tensor 的形状和 ``x`` 相同。如果 ``dtype`` 为 None，则输出 Tensor 的数据类型与 ``x`` 相同。
    - **low** (int，可选) - 要生成的随机值范围的下限，``low`` 包含在范围中。当 ``high`` 为 None 时，均匀采样的区间为[0, ``low``)。默认值为 0。
    - **high** (int，可选) - 要生成的随机值范围的上限，``high`` 不包含在范围中。默认值为 None，此时范围是[0, ``low``)。
    - **dtype** (str|np.dtype，可选) - 输出 Tensor 的数据类型，支持 bool，int32，int64，float16，float32，float64。当该参数值为 None 时，输出 Tensor 的数据类型与输入 Tensor 的数据类型一致。默认值为 None。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::
    Tensor：从区间[``low``，``high``)内均匀分布采样的随机 Tensor，形状为 ``x.shape``，数据类型为 ``dtype``。

代码示例
:::::::::::

COPY-FROM: paddle.randint_like
