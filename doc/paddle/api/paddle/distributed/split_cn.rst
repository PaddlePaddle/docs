.. _cn_api_distributed_split:

split
-------------------------------


.. py:function:: paddle.distributed.split(x, size, operatiion, axis=0, num_partitions=1, gather_out=True, param_attr=None, bias_attr=None, name=None)

切分指定操作的参数到多个设备，并且并行计算得到结果。

当前，支持一下三种情形。

情形1：并行Embedding
    Embedding操作的参数是个NxM的矩阵，行数为N，列数为M。并行Embedding情形下，参数切分到num_partitions个设备，每个设备上的参数是N/num_partitions行、M列的矩阵。

情形2：行并行Linear
    Linear操作的参数是个NxM的矩阵，行数为N，列数为M。行并行Linear情形下，参数切分到num_partitions个设备，每个设备上的参数是N/num_partitions行、M列的矩阵。

情形3：列并行Linear
    Linear操作的参数是个NxM的矩阵，行数为N，列数为M。列并行Linear情形下，参数切分到num_partitions个设备，每个设备上的参数是N行、M/num_partitions列的矩阵。

参数
:::::::::
    - x (Tensor) - 输入Tensor。Tensor的数据类型为：float16、float32、float64、int32、int64。
    - size (list|tuple) - 指定参数形状的列表或元组，包含2个元素。
    - operation (str) - 指定操作名称，当前支持的操作名称为'embedding'或'linear'。
    - axis (int，可选) - 指定沿哪个维度切分参数。
    - num_partitions (int，可选) - 指定参数的划分数。
    - gather_out (bool，可选) - 是否聚合所有设备的计算结果。
    - param_attr (ParamAttr，可选) - 指定参数的属性。
    - bias_attr (ParamAttr，可选) - 指定偏置的属性。
    - name (str，可选) - 默认值为None，通常用户不需要设置该属性。更多信息请参考 :ref:`api_guide_Name` 。

返回
:::::::::
Tensor

代码示例
:::::::::
.. code-block:: python

        import numpy
        import paddle
        from paddle.distributed import init_parallel_env

        paddle.set_device('gpu:%d'%paddle.distributed.ParallelEnv().dev_id)
        init_parallel_env()
        np_data = numpy.random.randint(0, 8, (10,4))
        data = paddle.to_tensor(np_data)
        emb_out = padle.distributed.split(
            data,
            (8, 8),
            operation="embedding",
            num_partitions=2)
