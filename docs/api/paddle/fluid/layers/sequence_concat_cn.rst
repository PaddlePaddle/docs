.. _cn_api_fluid_layers_sequence_concat:

sequence_concat
-------------------------------


.. py:function:: paddle.fluid.layers.sequence_concat(input, name=None)




**注意：该OP的输入只能是LoDTensor，如果您需要处理的输入是Tensor类型，请使用concat函数（fluid.layers.** :ref:`cn_api_fluid_layers_concat` **）。**

**该OP仅支持LoDTensor**，通过LoDTensor的LoD信息将输入的多个LoDTensor进行连接（concat），输出连接后的LoDTensor。

::

    input是由多个LoDTensor组成的list：
        input = [x1, x2]
    其中：
        x1.lod = [[0, 3, 5]]
        x1.data = [[1], [2], [3], [4], [5]]
        x1.shape = [5, 1]

        x2.lod = [[0, 2, 4]]
        x2.data = [[6], [7], [8], [9]]
        x2.shape = [4, 1]
    且必须满足：len(x1.lod[0]) == len(x2.lod[0])
    
    输出为LoDTensor：
        out.lod = [[0, 3+2, 5+4]]
        out.data = [[1], [2], [3], [6], [7], [4], [5], [8], [9]]
        out.shape = [9, 1]


参数
::::::::::::

        - **input** (list of Variable) – 多个LoDTensor组成的list，要求每个输入LoDTensor的LoD长度必须一致。数据类型为float32，float64或int64。
        - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
 输出连接后的LoDTensor，数据类型和输入一致。

返回类型
::::::::::::
 Variable


代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.sequence_concat