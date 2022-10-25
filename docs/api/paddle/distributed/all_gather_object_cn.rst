.. _cn_api_distributed_all_gather_object:

all_gather_object
-------------------------------


.. py:function:: paddle.distributed.all_gather_object(object_list, object, group=0)

进程组内所有进程指定的 picklable 对象进行聚合操作，并返回给所有进程聚合的结果。和 all_gather 类似，但可以传入自定义的 python 对象。

.. warning::
  该 API 只支持动态图模式。

参数
:::::::::
    - **object_list** (list) - 操作的输出 Object 列表。
    - **object** (Any) - 操作的输入 Object，需要保证输入自定义的 Object 是 picklable 的。
    - **group** (int，可选) - 工作的进程组编号，默认为 0。

返回
:::::::::
无

代码示例
:::::::::
COPY-FROM: paddle.distributed.all_gather_object
