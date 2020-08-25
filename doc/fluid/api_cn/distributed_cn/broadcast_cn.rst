.. _cn_api_distributed_broadcast:

broadcast
-------------------------------


.. py:function:: paddle.distributed.broadcast(tensor, src, group=0)

广播一个Tensor给其他所有进程

参数
:::::::::
    - tensor (Tensor) - 如果当前进程编号是源，那么这个Tensor变量将被发送给其他进程，否则这个Tensor将接收源发送过来的数据。Tensor的数据类型为：float32、float64、int32、int64。
    - src (int) - 发送源的进程编号。
    - group (int，可选) - 工作的进程组编号，默认为0。

返回
:::::::::
无

代码示例
:::::::::
.. code-block:: python

        import paddle
        import paddle.fluid as fluid
        from fluid.dygraph.parallel import prepare_context

        paddle.disable_static()
        place = fluid.CUDAPlace(fluid.dygraph.ParallelEnv().dev_id)
        with fluid.dygraph.guard(place=place):
            prepare_context()
            if fluid.dygraph.ParallelEnv().local_rank == 0:
                np_data = np.array([[4, 5, 6], [4, 5, 6]])
            else:
                np_data = np.array([[1, 2, 3], [1, 2, 3]])
            data = paddle.to_tensor(np_data)
            paddle.distributed.broadcast(data, 1)
            out = data.numpy()
            # [[1, 2, 3], [1, 2, 3]]


