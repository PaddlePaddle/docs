.. _cn_api_fluid_layers_eye:

eye
-------------------------------

.. py:function:: paddle.fluid.layers.eye(num_rows, num_columns=None, batch_shape=None, dtype='float32', name=None)


该OP用来构建二维Tensor，或一个批次的二维Tensor。

参数：
    - **num_rows** (int) - 该批次二维Tensor的行数，数据类型为非负int32。
    - **num_columns** (int, 可选) - 该批次二维Tensor的列数，数据类型为非负int32。若为None，则默认等于num_rows。
    - **batch_shape** (list(int), 可选) - 如若提供，则返回Tensor的主批次维度将为batch_shape。
    - **dtype** (np.dtype|core.VarDesc.VarType|str，可选) - 返回Tensor的数据类型，可为int32，int64，float16，float32，float64，默认数据类型为float32。
    - **name** (str) – 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None。
    
返回： ``shape`` 为batch_shape + [num_rows, num_columns]的Tensor。


**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    data = fluid.layers.eye(3, dtype='int32')
    # [[1, 0, 0]
    #  [0, 1, 0]
    #  [0, 0, 1]]

    data = fluid.layers.eye(2, 3, dtype='int32')
    # [[1, 0, 0]
    #  [0, 1, 0]]

    data = fluid.layers.eye(2, batch_shape=[3])
    # Construct a batch of 3 identity tensors, each 2 x 2.
    # data[i, :, :] is a 2 x 2 identity tensor, i = 0, 1, 2.






