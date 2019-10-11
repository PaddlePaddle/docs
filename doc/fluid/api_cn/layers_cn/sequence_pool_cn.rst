.. _cn_api_fluid_layers_sequence_pool:

sequence_pool
-------------------------------

.. py:function:: paddle.fluid.layers.sequence_pool(input, pool_type, is_test=False, pad_value=0.0)

**注意：该OP的输入只能是LoDTensor，如果您需要处理的输入是Tensor类型，请使用pool2d函数（fluid.layers.** :ref:`cn_api_fluid_layers_pool2d` **）。**

该OP **仅支持LoDTensor类型的输入** ，将对输入的LoDTensor进行指定方式的池化（pooling）操作。通过指定pool_type参数，将输入的每个序列（sequence）在最后一层lod_level上或时间步（time-step）上对特征进行诸如sum、average、sqrt等池化操作。

支持六种pool_type:

- **average**: :math:`Out[i] = \frac{\sum_{i}X_{i}}{N}`
- **sum**: :math:`Out[i] = \sum _{j}X_{ij}`
- **sqrt**: :math:`Out[i] = \frac{ \sum _{j}X_{ij}}{\sqrt{len(\sqrt{X_{i}})}}`
- **max**: :math:`Out[i] = max(X_{i})`
- **last**: :math:`Out[i] = X_{N\_i}`
- **first**: :math:`Out[i] = X_{0}`

其中 ``N_i`` 为待池化第i个输入序列的长度。

::

    Case 1:

        input是1-level的LoDTensor, 且pad_value = 0.0:
            input.lod = [[0, 2, 5, 7, 7]]
            input.data = [[1.], [3.], [2.], [4.], [6.], [5.], [1.]]
            input.shape = [7, 1]
        输出为LoDTensor：
            out.shape = [4, 1]
            其中 out.shape[0] == len(x.lod[-1]) == 4
        对于不同的pool_type：
            average: out.data = [[2.], [4.], [3.], [0.0]], where 2.=(1. + 3.)/2, 4.=(2. + 4. + 6.)/3, 3.=(5. + 1.)/2
            sum    : out.data = [[4.], [12.], [6.], [0.0]], where 4.=1. + 3., 12.=2. + 4. + 6., 6.=5. + 1.
            sqrt   : out.data = [[2.82], [6.93], [4.24], [0.0]], where 2.82=(1. + 3.)/sqrt(2), 6.93=(2. + 4. + 6.)/sqrt(3), 4.24=(5. + 1.)/sqrt(2)
            max    : out.data = [[3.], [6.], [5.], [0.0]], where 3.=max(1., 3.), 6.=max(2., 4., 6.), 5.=max(5., 1.)
            last   : out.data = [[3.], [6.], [1.], [0.0]], where 3.=last(1., 3.), 6.=last(2., 4., 6.), 1.=last(5., 1.)
            first  : out.data = [[1.], [2.], [5.], [0.0]], where 1.=first(1., 3.), 2.=first(2., 4., 6.), 5.=first(5., 1.)
        
        上述out.data中的最后一个[0.0]均为填充的数据。

    Case 2:
    
        input是2-level的LoDTensor, 包含3个长度分别为[2, 0, 3]的序列，其中中间的0表示序列为空。
        第一个长度为2的序列包含2个长度分别为[1, 2]的子序列；
        最后一个长度为3的序列包含3个长度分别为[1, 0, 3]的子序列。
            input.lod = [[0, 2, 2, 5], [0, 1, 3, 4, 4, 7]]
            input.data = [[1.], [3.], [2.], [4.], [6.], [5.], [1.]]
            input.shape = [7, 1]
        
        以pool_type取值为sum为例，将根据最后一层的lod信息[0, 1, 3, 4, 4, 7]进行池化操作，且pad_value = 0.0
        输出为LoDTensor：
            out.shape= [5, 1]
            out.lod = [[0, 2, 2, 5]]
            其中 out.shape[0] == len(x.lod[-1]) == 5
            sum: out.data = [[1.], [5.], [4.], [0.0], [12.]]
            where 1.=1., 5.=3. + 2., 4.=4., 0.0=pad_value, 12.=6. + 5. + 1.


参数：
    - **input** (Variable) - 类型为LoDTensor的输入序列，仅支持lod_level不超过2的LoDTensor，数据类型为float32。
    - **pool_type** (str) - 池化类型，支持average，sum，sqrt，max，last和first池化操作。
    - **is_test** (bool) - 仅在pool_type取值为max时生效。当is_test为False时，则在池化操作过程中会创建maxIndex临时Tenosr，以记录最大特征值对应的索引信息，用于训练阶段的反向梯度计算。默认为False。
    - **pad_value** (float) - 用于填充输入序列为空时的池化结果，默认为0.0。

返回：经过指定类型池化后的LoDTensor，数据类型为float32。

返回类型：Variable

**代码示例**:

.. code-block:: python

    import paddle.fluid as fluid

    x = fluid.layers.data(name='x', shape=[7, 1], append_batch_size=False,
                 dtype='float32', lod_level=1)
    avg_x = fluid.layers.sequence_pool(input=x, pool_type='average')
    sum_x = fluid.layers.sequence_pool(input=x, pool_type='sum')
    sqrt_x = fluid.layers.sequence_pool(input=x, pool_type='sqrt')
    max_x = fluid.layers.sequence_pool(input=x, pool_type='max')
    last_x = fluid.layers.sequence_pool(input=x, pool_type='last')
    first_x = fluid.layers.sequence_pool(input=x, pool_type='first')









