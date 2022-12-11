.. _cn_api_fluid_layers_sequence_last_step:

sequence_last_step
-------------------------------


.. py:function:: paddle.fluid.layers.sequence_last_step(input)




该OP **仅支持LoDTensor类型的输入**，将对输入的LoDTensor，在最后一层lod_level上，选取其每个序列（sequence）的最后一个时间步（time-step）的特征向量作为池化后的输出向量。

::

    Case 1:

        input是1-level的LoDTensor:
            input.lod = [[0, 2, 5, 7]]
            input.data = [[1.], [3.], [2.], [4.], [6.], [5.], [1.]]
            input.shape = [7, 1]

        输出为LoDTensor:
            out.shape = [3, 1]
            且 out.shape[0] == len(x.lod[-1]) == 3

            out.data = [[3.], [6.], [1.]], where 3.=last(1., 3.), 6.=last(2., 4., 6.), 1.=last(5., 1.)

    Case 2:
    
        input是2-level的LoDTensor，包含3个长度分别为[2, 0, 3]的序列，其中中间的0表示序列为空。
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
            out.data = [[1.], [2.], [4.], [0.0], [1.]]
            where 1.=last(1.), 2.=last(3., 2.), 4.=last(4.), 0.0 = pad_value, 1=last(6., 5., 1.)

参数
::::::::::::
**input** (Variable)- 类型为LoDTensor的输入序列，仅支持lod_level不超过2的LoDTensor，数据类型为float32。

返回
::::::::::::
每个输入序列中的最后一步特征向量组成的LoDTensor，数据类型为float32。

返回类型
::::::::::::
Variable

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.sequence_last_step