.. _cn_api_fluid_layers_eye:

eye
-------------------------------

.. py:function:: paddle.fluid.layers.eye(num_rows, num_columns=None, batch_shape=None, dtype='float32', name=None)


该 OP 用来构建二维 Tensor，或一个批次的二维 Tensor。

参数
::::::::::::

    - **num_rows** (int) - 该批次二维 Tensor 的行数，数据类型为非负 int32。
    - **num_columns** (int，可选) - 该批次二维 Tensor 的列数，数据类型为非负 int32。若为 None，则默认等于 num_rows。
    - **batch_shape** (list(int)，可选) - 如若提供，则返回 Tensor 的主批次维度将为 batch_shape。
    - **dtype** (np.dtype|core.VarDesc.VarType|str，可选) - 返回 Tensor 的数据类型，可为 int32，int64，float16，float32，float64，默认数据类型为 float32。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
 ``shape`` 为 batch_shape + [num_rows, num_columns]的 Tensor。


代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.eye
