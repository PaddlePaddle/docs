.. _cn_api_fluid_layers_sequence_pool:

sequence_pool
-------------------------------

.. py:function:: paddle.fluid.layers.sequence_pool(input, pool_type, is_test=False, pad_value=0.0)

该函数为序列的池化添加操作符。将每个实例的所有时间步数特征池化，并用参数中提到的pool_type将特征运用到输入到首部。

支持四种pool_type:

- **average**: :math:`Out[i] = \frac{\sum_{i}X_{i}}{N}`
- **sum**: :math:`Out[i] = \sum _{j}X_{ij}`
- **sqrt**: :math:`Out[i] = \frac{ \sum _{j}X_{ij}}{\sqrt{len(\sqrt{X_{i}})}}`
- **max**: :math:`Out[i] = max(X_{i})`

::


    x是一级LoDTensor且**pad_value** = 0.0:
        x.lod = [[2, 3, 2, 0]]
        x.data = [1, 3, 2, 4, 6, 5, 1]
        x.dims = [7, 1]
    输出为张量（Tensor）：
        out.dim = [4, 1]
        with condition len(x.lod[-1]) == out.dims[0]
    对于不同的pool_type：
        average: out.data = [2, 4, 3, 0.0], where 2=(1+3)/2, 4=(2+4+6)/3, 3=(5+1)/2
        sum    : out.data = [4, 12, 6, 0.0], where 4=1+3, 12=2+4+6, 6=5+1
        sqrt   : out.data = [2.82, 6.93, 4.24, 0.0], where 2.82=(1+3)/sqrt(2),
             6.93=(2+4+6)/sqrt(3), 4.24=(5+1)/sqrt(2)
        max    : out.data = [3, 6, 5, 0.0], where 3=max(1,3), 6=max(2,4,6), 5=max(5,1)
        last   : out.data = [3, 6, 1, 0.0], where 3=last(1,3), 6=last(2,4,6), 1=last(5,1)
        first  : out.data = [1, 2, 5, 0.0], where 1=first(1,3), 2=first(2,4,6), 5=first(5,1)
        
      且以上所有均满足0.0 = **pad_value**

参数：
    - **input** (variable) - 输入变量，为LoDTensor
    - **pool_type** (string) - 池化类型。支持average,sum,sqrt和max
    - **is_test** (bool, 默认为 False) - 用于区分训练模式和测试评分模式。默认为False。
    - **pad_value** (float) - 用于填充空输入序列的池化结果。

返回：sequence pooling 变量，类型为张量（Tensor)

**代码示例**:

.. code-block:: python

    import paddle.fluid as fluid

    x = fluid.layers.data(name='x', shape=[7, 1],
                 dtype='float32', lod_level=1)
    avg_x = fluid.layers.sequence_pool(input=x, pool_type='average')
    sum_x = fluid.layers.sequence_pool(input=x, pool_type='sum')
    sqrt_x = fluid.layers.sequence_pool(input=x, pool_type='sqrt')
    max_x = fluid.layers.sequence_pool(input=x, pool_type='max')
    last_x = fluid.layers.sequence_pool(input=x, pool_type='last')
    first_x = fluid.layers.sequence_pool(input=x, pool_type='first')









