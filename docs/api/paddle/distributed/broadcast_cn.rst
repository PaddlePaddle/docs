.. _cn_api_distributed_broadcast:

broadcast
-------------------------------


.. py:function:: paddle.distributed.broadcast(tensor, src, group=0)

广播一个Tensor给其他所有进程，如下图所示（https://images.nvidia.com/events/sc15/pdfs/NCCL-Woolley.pdf）。

.. image:: ./img/broadcast.png
  :width: 800
  :alt: broadcast
  :align: center

参数
:::::::::
    - tensor (Tensor) - 如果当前进程编号是源，那么这个Tensor变量将被发送给其他进程，否则这个Tensor将接收源发送过来的数据。Tensor的数据类型为：float16、float32、float64、int32、int64。
    - src (int) - 发送源的进程编号。
    - group (int，可选) - 工作的进程组编号，默认为0。

返回
:::::::::
无

代码示例
:::::::::
.. code-block:: python

        import numpy as np
        import paddle
        from paddle.distributed import init_parallel_env

        paddle.set_device('gpu:%d'%paddle.distributed.ParallelEnv().dev_id)
        init_parallel_env()
        if paddle.distributed.ParallelEnv().local_rank == 0:
            np_data = np.array([[4, 5, 6], [4, 5, 6]])
        else:
            np_data = np.array([[1, 2, 3], [1, 2, 3]])
        data = paddle.to_tensor(np_data)
        paddle.distributed.broadcast(data, 1)
        out = data.numpy()
        # [[1, 2, 3], [1, 2, 3]]
