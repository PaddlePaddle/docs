.. _cn_api_fluid_layers_uniform_random_batch_size_like:

uniform_random_batch_size_like
-------------------------------

.. py:function:: paddle.fluid.layers.uniform_random_batch_size_like(input, shape, dtype='float32', input_dim_idx=0, output_dim_idx=0, min=-1.0, max=1.0, seed=0)

该OP用输入Tensor指定维度的值替换返回Tensor指定维度的值，并使用从均匀分布中采样的随机值初始化Tensor。

::

    示例1:
         input =[[0.946741  , 0.1357001 , 0.38086128]]    # input.shape=[1,3]
         shape=[2,4]
    则：
         result=[[ 0.3443427 , -0.23056602,  0.3477049 ,  0.06139076]]    # result.shape=[1,4]

    示例2:
         input =[[0.946741  , 0.1357001 , 0.38086128]]     # input.shape=[1,3]
         shape=[2,4]
         input_dim_idx=1
         output_dim_idx=1
    则：
         result=[[-0.23133647, -0.84195036,  0.21441269],
                 [-0.08774924,  0.25605237, -0.09403259]]    # result.shape=[2,3]

参数：
        - **input** （Variable）- 输入Tensor，input_dim_idx将指定其维度用来替换返回Tensor的指定维度。
        - **shape** （list|tuple）- 设置返回Tensor的维度，其中output_dim_idx参数指定维度的值将被替代。数据类型为int。
        - **input_dim_idx** （int，可选）- 输入Tensor指定维度的索引，数据类型为int。默认值为0。
        - **output_dim_idx** （int，可选）- 返回Tensor指定维度的索引，数据类型为int。默认值为0。
        - **min** （float，可选）- 均匀随机的最小值，为闭区间。数据类型为float。默认值为 1.0。
        - **max** （float，可选）- 均匀随机的最大值，为开区间。数据类型为float。默认值为1.0。
        - **seed** （int，可选）- 用于生成样本的随机种子。0表示使用系统生成的种子，数据类型为int。注意如果seed不为0，则此算子将始终每次生成相同的随机数。默认值为0。
        - **dtype** （np.dtype | core.VarDesc.VarType | str） - 返回结果的数据类型：float32，float16，int等。

返回:      表示随机初始化结果的tensor，数据类型由dtype参数设置。

返回类型:        Variable


**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid
    import paddle.fluid.layers as layers
    
    
    input = layers.data(name="input", shape=[13, 11], dtype='float32')
    # examp 1:
    # input_dim_idx和output_dim_idx使用默认值 
    out1 = layers.uniform_random_batch_size_like(input, [3, 5]) 
    out1_shape = layers.shape(out1) # [13,5]
   
    # example 2:
    # input_dim_idx和output_dim_idx使用指定值
    out2=layers.uniform_random_batch_size_like(input, [3, 5], input_dim_idx=1, output_dim_idx=1)
    out2_shape = layers.shape(out2) # [3,11]        




