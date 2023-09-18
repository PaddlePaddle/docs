.. _cn_api_paddle_rand:

rand
----------------------

.. py:function:: paddle.rand(shape, dtype=None, name=None)

返回符合均匀分布的、范围在[0, 1)的 Tensor，形状为 ``shape``，数据类型为 ``dtype``。

参数
::::::::::
    - **shape** (list|tuple|Tensor) - 生成的随机 Tensor 的形状。如果 ``shape`` 是 list、tuple，则其中的元素可以是 int，或者是形状为[]且数据类型为 int32、int64 的 0-D Tensor。如果 ``shape`` 是 Tensor，则是数据类型为 int32、int64 的 1-D Tensor。
    - **dtype** (str|np.dtype，可选) - 输出 Tensor 的数据类型，支持 float32、float64。当该参数值为 None 时，输出 Tensor 的数据类型为 float32。使用全局默认dtype(详细信息请见 :ref:`get_default_dtype` )。默认值为 None。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::
    Tensor：符合均匀分布的范围为[0, 1)的随机 Tensor，形状为 ``shape``，数据类型为 ``dtype``。

示例代码
::::::::::

COPY-FROM: paddle.rand
