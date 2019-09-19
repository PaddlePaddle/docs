.. _cn_api_fluid_layers_reduce_mean:

reduce_mean
-------------------------------

.. py:function:: paddle.fluid.layers.reduce_mean(input, dim=None, keep_dim=False, name=None)

默认计算所有维度的平均值，当给定dim的值得时候计算给定维度上张量（Tensor）元素平均值。

参数：
          - **input** （Variable）：输入变量为Tensor或LoDTensor。
          - **dim** （list | int | None）：函数运算的维度。如果为None，则对输入的所有元素求平均值并返回单个元素的Tensor变量，dim的维度大小必须在  :math:`[−rank(input),rank(input))` 范围内。如果 :math:`dim [i] <0` ，则维度实际值为 :math:`rank+dim[i]` 。
          - **keep_dim** （bool | False）：是否在输出Tensor中保留减小的维度。除非 ``keep_dim`` 为true，否则结果张量rank大小会发生变化，对应操作维度会被移出。
          - **name** （str | None）：这一层的名称（可选）。如果设置为None，则将自动命名这一层。

返回：  在某些特定维度或者所有维度计算平均值之后的Tensor变量结果。

返回类型：  变量（Variable）

**代码示例**

..  code-block:: python

      import paddle.fluid as fluid
      # x是一个Tensor，元素如下:
      #    [[0.2, 0.3, 0.5, 0.9]
      #     [0.1, 0.2, 0.6, 0.7]]
      # 接下来的示例中，我们在每处函数调用后面都标注出了它的结果张量。
      x = fluid.layers.data(name='x', shape=[4, 2], dtype='float32')
      fluid.layers.reduce_mean(x)  # [0.4375]
      fluid.layers.reduce_mean(x, dim=0)  # [0.15, 0.25, 0.55, 0.8]
      fluid.layers.reduce_mean(x, dim=-1)  # [0.475, 0.4]
      fluid.layers.reduce_mean(x, dim=1, keep_dim=True)  # [[0.475], [0.4]]

      # y是一个shape为[2, 2, 2]的Tensor元素如下:
      #      [[[1.0, 2.0], [3.0, 4.0]],
      #      [[5.0, 6.0], [7.0, 8.0]]]
      # 接下来的示例中，我们在每处函数调用后面都标注出了它的结果张量。。
      y = fluid.layers.data(name='y', shape=[2, 2, 2], dtype='float32')
      fluid.layers.reduce_mean(y, dim=[1, 2]) # [2.5, 6.5]
      fluid.layers.reduce_mean(y, dim=[0, 1]) # [4.0, 5.0]










