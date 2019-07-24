.. _cn_api_fluid_layers_reduce_prod:

reduce_prod
-------------------------------

.. py:function:: paddle.fluid.layers.reduce_prod(input, dim=None, keep_dim=False, name=None)

计算给定维度上张量（Tensor）元素乘积。

参数：
          - **input** （Variable）：输入变量为Tensor或LoDTensor。
          - **dim** （list | int | None）：函数运算的维度。如果为None，则将输入的所有元素相乘并返回单个元素的Tensor变量，否则必须在  :math:`[−rank(input),rank(input)]` 范围内。如果 :math:`dim [i] <0` ，则维度将减小为 :math:`rank+dim[i]` 。
          - **keep_dim** （bool | False）：是否在输出Tensor中保留减小的维度。除非 ``keep_dim`` 为true，否则结果张量将比输入少一个维度。
          - **name** （str | None）：这一层的名称（可选）。如果设置为None，则将自动命名这一层。

返回：  运算、减少维度之后的Tensor变量。

返回类型：  变量（Variable）

**代码示例**

..  code-block:: python

      import paddle.fluid as fluid
      # x是一个Tensor，元素如下:
      #    [[0.2, 0.3, 0.5, 0.9]
      #     [0.1, 0.2, 0.6, 0.7]]
      # 接下来的示例中，我们在每处函数调用后面都标注出了它的结果张量。
      x = fluid.layers.data(name='x', shape=[4, 2], dtype='float32')
      fluid.layers.reduce_prod(x)  # [0.0002268]
      fluid.layers.reduce_prod(x, dim=0)  # [0.02, 0.06, 0.3, 0.63]
      fluid.layers.reduce_prod(x, dim=-1)  # [0.027, 0.0084]
      fluid.layers.reduce_prod(x, dim=1,
                               keep_dim=True)  # [[0.027], [0.0084]]

      # y 是一个shape为[2, 2, 2]的Tensor元素如下:
      #      [[[1.0, 2.0], [3.0, 4.0]],
      #      [[5.0, 6.0], [7.0, 8.0]]]
      # 接下来的示例中，我们在每处函数调用后面都标注出了它的结果张量。
      y = fluid.layers.data(name='y', shape=[2, 2, 2], dtype='float32')
      fluid.layers.reduce_prod(y, dim=[1, 2]) # [24.0, 1680.0]
      fluid.layers.reduce_prod(y, dim=[0, 1]) # [105.0, 384.0]










