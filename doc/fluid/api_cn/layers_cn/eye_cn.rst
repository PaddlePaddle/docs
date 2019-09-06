.. _cn_api_fluid_layers_eye:

eye
-------------------------------

.. py:function:: paddle.fluid.layers.eye(num_rows, num_columns=None, batch_shape=None, dtype='float32')

这个函数用来构建一个元素一样的向量，或一个向量batch。

参数：
    - **num_rows** (int) - 每一个batch向量的行数。
    - **num_columns** (int) - 每一个batch向量的列数，若为None，则等于num_rows。
    - **batch_shape** (list(int)) - 如若提供，则返回向量将会有一个此shape的主要的batch size。
    - **dtype** (string) - 'float32'|'int32'|...，返回向量的数据类型
    
返回：一个元素一样的向量，shape为batch_shape + [num_rows, num_columns]。

返回类型：变量（Variable）

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






