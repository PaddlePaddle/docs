.. _cn_api_paddle_zeros:

zeros
-------------------------------

.. py:function:: paddle.zeros(shape, dtype=None, name=None)



创建形状为 ``shape`` 、数据类型为 ``dtype`` 且值全为 0 的 Tensor。

.. note::
    别名支持: 参数名 ``size`` 可替代 ``shape``。
    ``shape`` 支持可变参数类型。
    使用实例：
        ``paddle.ones(1, 2, 3, dtype=paddle.float32)``
        ``paddle.ones(size=[1, 2, 3], dtype=paddle.float32)``

参数
::::::::::::

    - **shape** (tuple|list|Tensor) - 输出 Tensor 的形状，``shape`` 的数据类型为 int32 或者 int64，支持可变参数类型.
    - **size** - ``shape`` 的别名，行为完全一致。
    - **dtype** (np.dtype|str，可选) - 输出 Tensor 的数据类型，数据类型必须为 bool、float16、float32、float64、int32 或 int64。若为 None，数据类型为 float32，默认为 None。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
值全为 0 的 Tensor，数据类型和 ``dtype`` 定义的类型一致。


代码示例
::::::::::::

COPY-FROM: paddle.zeros
