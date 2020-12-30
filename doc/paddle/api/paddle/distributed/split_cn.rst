.. _cn_api_distributed_split:

split
-------------------------------


.. py:function:: paddle.distributed.split(x, size, operatiion, axis=0, num_partitions=1, gather_out=True, param_attr=None, bias_attr=None, name=None)

切分指定操作的参数到多个设备，并且并行计算得到结果。

当前，支持一下三种情形。

情形1：并行Embedding
    Embedding操作的参数是个NxM的矩阵，行数为N，列数为M。并行Embedding情形下，参数切分到num_partitions个设备，每个设备上的参数是 (N/num_partitions + 1)行、M列的矩阵。其中，最后一行作为padding idx。

    假设将NxM的参数矩阵切分到两个设备device_0和device_1。那么每个设置上的参数矩阵为(N/2+1)行和M列。device_0上，输入x中的值如果介于[0, N/2-1]，则其值保持不变；否则值变更为N/2，经过embedding映射为全0值。类似地，device_1上，输入x中的值V如果介于[N/2, N-1]之间，那么这些值将变更为(V-N/2)；否则，值变更为N/2，经过embedding映射为全0值。最后，使用all_reduce_sum操作汇聚各个卡上的结果。

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
