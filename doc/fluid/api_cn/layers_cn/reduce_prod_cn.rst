.. _cn_api_fluid_layers_reduce_prod:

reduce_prod
-------------------------------

.. py:function:: paddle.fluid.layers.reduce_prod(input, dim=None, keep_dim=False, name=None)

该OP是对指定维度上的Tensor元素进行求乘积运算，并输出相应的计算结果。

参数：
          - **input** （Variable）- 输入变量为多维Tensor或LoDTensor，支持数据类型为float32，float64，int32，int64。
          - **dim** （list | int ，可选）- 求乘积运算的维度。如果为None，则计算所有元素的乘积并返回包含单个元素的Tensor变量，否则必须在  :math:`[−rank(input),rank(input)]` 范围内。如果 :math:`dim [i] <0` ，则维度将变为 :math:`rank+dim[i]` ，默认值为None。
          - **keep_dim** （bool）- 是否在输出Tensor中保留减小的维度。如 keep_dim 为true，否则结果张量的维度将比输入张量小，默认值为False。
          - **name** （str ， 可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：  在指定dim上进行求乘积运算的Tensor，数据类型和输入数据类型一致。

返回类型：  变量（Variable）

**代码示例**

..  code-block:: python

      import paddle.fluid as fluid
      # x是一个Tensor，元素如下:
      #    [[0.2, 0.3, 0.5, 0.9]
      #     [0.1, 0.2, 0.6, 0.7]]
      # 接下来的示例中，我们在每处函数调用后面都标注出了它的结果张量。
      x = fluid.layers.data(name='x', shape=[2, 4], dtype='float32')
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










