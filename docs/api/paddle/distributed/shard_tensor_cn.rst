.. _cn_api_distributed_shard_tensor:

shard_tensor
-------------------------------

.. py:function:: paddle.distributed.shard_tensor(x, dist_attr=None)

为张量添加分布式属性。

参数
:::::::::
    - x (Tensor) - 待切分的张量。
    - dst (int) - 张量的分布式属性。可接受的属性如下:
        "process_mesh": 一个嵌套的列表，用来描述逻辑进程的网状拓扑结构。   
        "dims_mapping": 描述维度x和process_mesh之间映射的列表,x的i在process_mesh的process_mesh维度上被分割，
            其中-1表示张量维数没有分割。  
        process_mesh和dims_mapping都是可选的，用户可以根据需要指定。

返回
:::::::::
张量:用分布式属性标注的张量x。


代码示例
:::::::::
.. code-block:: python

    import paddle
    import paddle.distributed as dist

    paddle.enable_static()

    x = paddle.ones([4, 6])
    dist.shard_tensor(x, dist_attr={"process_mesh": [[0, 1], [2, 3]],
                                    "dims_mapping": [0, -1]})