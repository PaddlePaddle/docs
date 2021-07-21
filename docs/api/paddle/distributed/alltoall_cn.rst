.. _cn_api_distributed_alltoall:

alltoall
-------------------------------


.. py:function:: paddle.distributed.alltoall(in_tensor_list, out_tensor_list, group=None, use_calc_stream=True)

将in_tensor_list里面的tensors分发到所有参与的卡并将结果tensors汇总到out_tensor_list。


参数
:::::::::
    - in_tensor_list (list) - 包含所有输入Tensors的一个列表。在列表里面的所有元素都必须是一个Tensor，Tensor的数据类型必须是float16、float32、 float64、int32、int64。
    - out_tensor_list (Tensor) - 包含所有输出Tensors的一个列表。在列表里面的所有元素数据类型要和输入的Tensors数据类型一致。
    - group (Group, 可选) - new_group返回的Group实例，或者设置为None表示默认地全局组。默认值：None。
    - use_calc_stream (bool，可选) - 标识使用计算流还是通信流。默认值：True。

返回
:::::::::
无

代码示例
:::::::::
.. code-block:: python

        # required: distributed
        import numpy as np
        import paddle
        from paddle.distributed import init_parallel_env
        init_parallel_env()
        out_tensor_list = []
        if paddle.distributed.ParallelEnv().rank == 0:
            np_data1 = np.array([[1, 2, 3], [4, 5, 6]])
            np_data2 = np.array([[7, 8, 9], [10, 11, 12]])
        else:
            np_data1 = np.array([[13, 14, 15], [16, 17, 18]])
            np_data2 = np.array([[19, 20, 21], [22, 23, 24]])
        data1 = paddle.to_tensor(np_data1)
        data2 = paddle.to_tensor(np_data2)
        paddle.distributed.alltoall([data1, data2], out_tensor_list)
        # out for rank 0: [[[1, 2, 3], [4, 5, 6]], [[13, 14, 15], [16, 17, 18]]]
        # out for rank 1: [[[7, 8, 9], [10, 11, 12]], [[19, 20, 21], [22, 23, 24]]]

