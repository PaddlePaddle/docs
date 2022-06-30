.. _cn_api_distributed_scatter:

scatter
-------------------------------


.. py:function:: paddle.distributed.scatter(tensor, tensor_list=None, src=0, group=0)

进程组内指定进程源的tensor列表分发到其他所有进程中。
如下图所示，4个GPU分别开启4个进程，scatter的源选择为第0张卡，
经过scatter算子后，会将第0张卡的数据平均分到所有卡上。

.. image:: ./img/scatter.png
  :width: 800
  :alt: scatter
  :align: center

参数
:::::::::
    - tensor (Tensor) - 操作的输出Tensor。Tensor的数据类型为：float16、float32、float64、int32、int64。
    - tensor_list (list，可选) - 操作的输入Tensor列表，默认为None。列表中的每个元素均为Tensor，每个Tensor的数据类型为：float16、float32、float64、int32、int64。
    - src (int，可选) - 操作的源进程号，该进程号的Tensor列表将分发到其他进程中。默认为0。
    - group (int，可选) - 工作的进程组编号，默认为0。

返回
:::::::::
无

代码示例
:::::::::
COPY-FROM: paddle.distributed.scatter