.. _cn_api_fluid_layers_transpose:

transpose
-------------------------------

.. py:function:: paddle.fluid.layers.transpose(x,perm,name=None)

根据perm对输入矩阵维度进行重排。

返回张量（tensor）的第i维对应输入维度矩阵的perm[i]。

参数：
    - **x** (Variable) - 输入张量（Tensor)
    - **perm** (list) - 输入维度矩阵的转置
    - **name** (str) - 该层名称（可选）

返回： 转置后的张量（Tensor）

返回类型：变量（Variable）

**代码示例**:

.. code-block:: python

    # 请使用 append_batch_size=False 来避免
    # 在数据张量中添加多余的batch大小维度
    import paddle.fluid as fluid
    x = fluid.layers.data(name='x', shape=[5, 10, 15],
                    dtype='float32', append_batch_size=False)
    x_transposed = fluid.layers.transpose(x, perm=[1, 0, 2])




