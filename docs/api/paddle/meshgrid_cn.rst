.. _cn_api_paddle_tensor_meshgrid:

meshgrid
-------------------------------

.. py:function:: paddle.meshgrid(*args, **kargs)




该OP的输入是张量或者包含张量的列表, 包含 k 个一维张量，对每个张量做扩充操作，输出 k 个 k 维张量。

参数：
         - \* **args** （Tensor|Tensor数组）- 输入变量为 k 个一维张量，形状分别为(N1,), (N2,), ..., (Nk, )。支持数据类型为float32，float64，int32，int64。
         - ** **kargs** （可选）- 目前只接受name参数（str），具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回： 
k 个 k 维张量，每个张量的形状均为(N1, N2, ..., Nk)。

返回类型：  Tensor，k个Tensor，每个Tensor的形状分别为N1, N2, ... , Nk。

**代码示例**



..  code-block:: python

    import paddle

    x = paddle.randint(low=0, high=100, shape=[100])
    y = paddle.randint(low=0, high=100, shape=[200])

    grid_x, grid_y = paddle.meshgrid(x, y)

    print(grid_x.shape)
    print(grid_y.shape)

    #the shape of res_1 is (100, 200)
    #the shape of res_2 is (100, 200)  
