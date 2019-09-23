.. _cn_api_fluid_layers_reduce_max:

reduce_max
-------------------------------

.. py:function:: paddle.fluid.layers.reduce_max(input, dim=None, keep_dim=False, name=None)

该OP是对指定维度上的Tensor元素求最大值运算，并输出相应的计算结果。

参数：
          - **input** （Variable）- 输入变量为多维Tensor或LoDTensor。
          - **dim** （list | int ，可选）— 求最大值运算的维度。如果为None，则计算所有元素的最大值并返回包含单个元素的Tensor变量，否则必须在  :math:`[−rank(input),rank(input)]` 范围内。如果 :math:`dim [i] <0` ，则维度将减小为 :math:`rank+dim[i]` ，默认值为None。
          - **keep_dim** （bool）- 是否在输出Tensor中保留减小的维度。如 keep_dim 为true，否则结果张量的维度将比输入张量小，默认值为False。
          - **name** （str ， 可选）— 这一层的名称。如果设置为None，则将自动命名这一层。默认值为None。

返回：  在指定dim上进行求最大值运算的Tensor，数据类型和输入数据类型一致。

返回类型：  变量（Variable）

**代码示例**

..  code-block:: python

      import paddle.fluid as fluid
      # x是一个Tensor，元素如下:
      #    [[0.2, 0.3, 0.5, 0.9]
      #     [0.1, 0.2, 0.6, 0.7]]
      # 接下来的示例中，我们在每处函数调用后面都标注出了它的结果张量。
      x = fluid.layers.data(name='x', shape=[2, 4], dtype='float32')
      fluid.layers.reduce_max(x)  # [0.9]
      fluid.layers.reduce_max(x, dim=0)  # [0.2, 0.3, 0.6, 0.9]
      fluid.layers.reduce_max(x, dim=-1)  # [0.9, 0.7]
      fluid.layers.reduce_max(x, dim=1, keep_dim=True)  # [[0.9], [0.7]]

      # y是一个shape为[2, 2, 2]的Tensor，元素如下:
      #      [[[1.0, 2.0], [3.0, 4.0]],
      #      [[5.0, 6.0], [7.0, 8.0]]]
      # 接下来的示例中，我们在每处函数调用后面都标注出了它的结果张量。
      y = fluid.layers.data(name='y', shape=[2, 2, 2], dtype='float32')
      fluid.layers.reduce_max(y, dim=[1, 2]) # [4.0, 8.0]
      fluid.layers.reduce_max(y, dim=[0, 1]) # [7.0, 8.0]










