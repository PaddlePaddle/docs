
.. _cn_api_paddle_tensor_meshgrid:

meshgrid
-------------------------------

.. py:function:: paddle.tensor.meshgrid(*args, **kargs)

:alias_main: paddle.meshgrid
:alias: paddle.meshgrid, paddle.tensor.meshgrid, paddle.tensor.creation.meshgrid



该OP的输入是张量或者包含张量的列表, 包含 k 个一维张量，对每个张量做扩充操作，输出 k 个 k 维张量。

参数：
         - \* **args** （Variable|Variable数组）- 输入变量为 k 个一维张量，形状分别为(N1,), (N2,), ..., (Nk, )。支持数据类型为float32，float64，int32，int64。
         - ** **kargs** （可选）- 目前只接受name参数（str），具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回： 
k 个 k 维张量，每个张量的形状均为(N1, N2, ..., Nk)。

返回类型：  变量（Variable）

**代码示例**



..  code-block:: python

    #动态图示例
    import paddle
    import numpy as np
    
    paddle.enable_imperative()

    input_3 = np.random.randint(0, 100, [100, ]).astype('int32')
    input_4 = np.random.randint(0, 100, [200, ]).astype('int32')
    tensor_3 = paddle.imperative.to_variable(input_3)
    tensor_4 = paddle.imperative.to_variable(input_4)
    grid_x, grid_y = paddle.tensor.meshgrid(tensor_3, tensor_4)
    #the shape of grid_x is (100, 200)
    #the shape of grid_y is (100, 200)    
