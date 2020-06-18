.. _cn_api_fluid_layers_sequence_first_step:

sequence_first_step
-------------------------------


.. py:function:: paddle.fluid.layers.sequence_first_step(input)

:api_attr: 声明式编程模式（静态图)



该OP **仅支持LoDTensor类型的输入** ，将对输入的LoDTensor，在最后一层lod_level上，选取其每个序列（sequence）的第一个时间步（time_step）的特征向量作为池化后的输出向量。

::

    Case 1:

      input是1-level LoDTensor:
        input.lod = [[0, 2, 5, 7]]
        input.data = [[1.], [3.], [2.], [4.], [6.], [5.], [1.]]
        input.shape = [7, 1]

      输出为LoDTensor:
        out.shape = [3, 1]
        且 out.shape[0] == len(x.lod[-1]) == 3
        out.data = [[1.], [2.], [5.]], where 1.=first(1., 3.), 2.=first(2., 4., 6.), 5.=first(5., 1.)

    Case 2:
    
      input是2-level的LoDTensor, 包含3个长度分别为[2, 0, 3]的序列，其中中间的0表示序列为空。
      第一个长度为2的序列包含2个长度分别为[1, 2]的子序列；
      最后一个长度为3的序列包含3个长度分别为[1, 0, 3]的子序列。
          input.lod = [[0, 2, 2, 5], [0, 1, 3, 4, 4, 7]]
          input.data = [[1.], [3.], [2.], [4.], [6.], [5.], [1.]]
          input.shape = [7, 1]
      
      将根据最后一层的lod信息[0, 1, 3, 4, 4, 7]进行池化操作，且pad_value = 0.0
      输出为LoDTensor：
          out.shape= [5, 1]
          out.lod = [[0, 2, 2, 5]]
          其中 out.shape[0] == len(x.lod[-1]) == 5
          out.data = [[1.], [3.], [4.], [0.0], [6.]]
          where 1.=first(1.), 3.=first(3., 2.), 4.=first(4.), 0.0 = pad_value, 6.=first(6., 5., 1.)

参数：**input** (Variable)- 类型为LoDTensor的输入序列，仅支持lod_level不超过2的LoDTensor，数据类型为float32。

返回：每个输入序列中的第一个step的特征向量组成的LoDTensor，数据类型为float32。

返回类型：Variable

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    x = fluid.layers.data(name='x', shape=[7, 1], append_batch_size=False,
                 dtype='float32', lod_level=1)
    x_first_step = fluid.layers.sequence_first_step(input=x)









