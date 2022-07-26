.. _cn_api_distributed_all_gather_object:

all_gather_object
-------------------------------


.. py:function:: paddle.distributed.all_gather_object(object_list, object, group=0)

进程组内所有进程指定的picklable对象进行聚合操作，并返回给所有进程聚合的结果。和all_gather类似，但可以传入自定义的python对象。

参数
:::::::::
    - object_list (list) - 操作的输出Object列表。
    - object (Any) - 操作的输入Object，需要保证输入自定义的Object是picklable的。
    - group (int，可选) - 工作的进程组编号，默认为0。

返回
:::::::::
无

代码示例
:::::::::
COPY-FROM: paddle.distributed.all_gather_object
