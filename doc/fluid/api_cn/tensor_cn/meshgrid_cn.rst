
.. _cn_api_fluid_layers_meshgrid:

meshgrid
-------------------------------

.. py:function:: fluid.layers.meshgrid(input, name=None)

该OP的输入是tensor list, 包含 k 个一维Tensor，对每个Tensor做扩充操作，输出 k 个 k 维tensor。

参数：
         - **input** （Variable）- 输入变量为 k 个一维Tensor，形状分别为(N1,), (N2,), ..., (Nk, )。支持数据类型为float32，float64，int32，int64。
         - **name** （str， 可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回： 
k 个 k 维Tensor，每个Tensor的形状均为(N1, N2, ..., Nk)。

返回类型：  变量（Variable）

**代码示例**

..  code-block:: python

    #静态图示例
    import paddle
    import paddle.fluid as fluid
    import numpy as np
    x = fluid.data(name='x', shape=[100], dtype='int32')
    y = fluid.data(name='y', shape=[200], dtype='int32')
    input_1 = np.random.randint(0, 100, [100, ]).astype('int32')
    input_2 = np.random.randint(0, 100, [200, ]).astype('int32')
    exe = fluid.Executor(place=fluid.CPUPlace())
    grid_x, grid_y = fluid.layers.meshgrid([x, y])
    res_1, res_2 = exe.run(fluid.default_main_program(),
                            feed={'x': input_1,
                                  'y': input_2},
                            fetch_list=[grid_x, grid_y])
     
    #the shape of res_1 is (100, 200)
    #the shape of res_2 is (100, 200)


..  code-block:: python

    #动态图示例
    import paddle
    import paddle.fluid as fluid
    import numpy as np
    input_3 = np.random.randint(0, 100, [100, ]).astype('int32')
    input_4 = np.random.randint(0, 100, [200, ]).astype('int32')
    with fluid.dygraph.guard():
        tensor_3 = fluid.dygraph.to_variable(input_3)
        tensor_4 = fluid.dygraph.to_variable(input_4)
        grid_x, grid_y = fluid.layers.meshgrid([tensor_3, tensor_4])
    #the shape of grid_x is (100, 200)
    #the shape of grid_y is (100, 200)    
