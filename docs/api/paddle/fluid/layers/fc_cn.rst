.. _cn_api_fluid_layers_fc:

fc
-------------------------------


.. py:function::  paddle.fluid.layers.fc(input, size, num_flatten_dims=1, param_attr=None, bias_attr=None, act=None, name=None)





**全连接层**

该OP将在神经网络中构建一个全连接层。其输入可以是一个Tensor（或LoDTensor）或多个Tensor（或LoDTensor）组成的list（详见参数说明），该OP会为每个输入的Tensor创建一个权重（weights）变量，即一个从每个输入单元到每个输出单元的全连接权重矩阵。FC层将每个输入Tensor和其对应的权重(weights)相乘得到shape为 :math:`[M, size]` 输出Tensor，其中 ``M`` 为batch_size大小。如果有多个输入Tensor，则多个shape为 :math:`[M, size]` 的Tensor计算结果会被累加起来，作为最终输出。如果 ``bias_attr`` 非空，则会创建一个偏置变量（bias variable），并把它累加到输出结果中。如果 ``act`` 非空，将会在输出结果上应用相应的激活函数。

当输入为单个Tensor（或LoDTensor）：

.. math::

        \\Out = Act({XW + b})\\



当输入为多个Tensor（或LoDTensor）组成的list时：

.. math::

        \\Out=Act(\sum^{N-1}_{i=0}X_iW_i+b) \\


上述等式中：
  - :math:`N`：输入的数目，如果输入是Tensor列表，N等于len(input)
  - :math:`X_i`：第i个输入的Tensor
  - :math:`W_i`：对应第i个输入张量的第i个权重矩阵
  - :math:`b`：该层创建的bias参数
  - :math:`Act` ：activation function(激活函数)
  - :math:`Out`：输出Tensor

::
            
        Case 1： 
            给定单个输入Tensor data_1，且num_flatten_dims = 2:
                data_1.data = [[[0.1, 0.2],
                               [0.3, 0.4]]]
                data_1.shape = (1, 2, 2) # 1是batch_size

                out = fluid.layers.fc(input=data_1, size=1， num_flatten_dims=2)

          则输出为：
                out.data = [[0.83234344], [0.34936576]]
                out.shape = (1, 2, 1)


        Case 2: 
            给定多个Tensor组成的list:
                data_1.data = [[[0.1, 0.2],
                               [0.3, 0.4]]]
                data_1.shape = (1, 2, 2) # 1 是 batch_size

                data_2 = [[[0.1, 0.2, 0.3]]]
                data_2.shape = (1, 1, 3)

                out = fluid.layers.fc(input=[data_1, data_2], size=2)

            则输出为：
                out.data = [[0.18669507, 0.1893476]]
                out.shape = (1, 2)


参数
::::::::::::

  - **input** (Variable|list of Variable) – 维度为 :math:`[N_1, N_2, ..., N_k]` 的多维Tensor（或LoDTensor）或由多个Tensor（或LoDTensor）组成的list，输入Tensor的shape至少是2。数据类型为float32或float64。
  - **size** (int) – 全连接层输出单元的数目，即输出Tensor（或LoDTensor）特征维度。
  - **num_flatten_dims** (int) – 输入可以接受维度大于2的Tensor。在计算时，输入首先会被扁平化（flatten）为一个二维矩阵，之后再与权重(weights)相乘。参数 ``num_flatten_dims`` 决定了输入Tensor的flatten方式：前 ``num_flatten_dims`` (包含边界，从1开始数) 个维度会被扁平化为二维矩阵的第一维 (即为矩阵的高)，剩下的 :math:`rank(X) - num\_flatten\_dims` 维被扁平化为二维矩阵的第二维 (即矩阵的宽)。例如，假设X是一个五维的Tensor，其shape为(2, 3, 4, 5, 6)，若 :math:`num\_flatten\_dims = 3`，则扁平化的矩阵shape为：:math:`(2 x 3 x 4, 5 x 6) = (24, 30)`，最终输出Tensor的shape为 :math:`(2, 3, 4, size)`。默认为1。
  - **param_attr** (ParamAttr) – 指定权重参数属性的对象。默认值为None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
  - **bias_attr** (ParamAttr) – 指定偏置参数属性的对象。默认值为None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
  - **act** (str) – 应用于输出上的激活函数，如tanh、softmax、sigmoid，relu等，支持列表请参考 :ref:`api_guide_activations`，默认值为None。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
::::::::::::
经过全连接层计算后的Tensor或LoDTensor，数据类型与input类型一致。

返回类型
::::::::::::
 Variable

弹出异常：``ValueError`` - 如果输入Tensor（或LoDTensor）的维度小于2

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.fc