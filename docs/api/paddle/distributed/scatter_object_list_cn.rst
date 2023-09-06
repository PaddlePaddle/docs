.. _cn_api_distributed_scatter_object_list:

scatter_object_list
-------------------------------


.. py:function:: paddle.distributed.scatter_object_list(out_object_list, in_object_list, src=0, group=None)

将一组来自指定进程的 picklable 对象分发到每个进程
过程与 ``scatter`` 类似，但可以传入自定义的 python 对象。

.. note::
  该 API 只支持动态图模式。

参数
:::::::::
    - **out_object_list** (List[Any]) - 用于接收数据的 object 列表。
    - **in_object_list** (List[Any]，可选) - 将被分发的 object 列表。默认为 None，因为 rank != src 的进程上的该参数将被忽略。
    - **src** (int，可选) - 目标进程的 rank，该进程的 object 列表将被分发到其他进程中。默认为 0，即分发 rank=0 的进程上的 object 列表。
    - **group** (Group，可选) - 执行该操作的进程组实例（通过 ``new_group`` 创建）。默认为 None，即使用全局默认进程组。

返回
:::::::::
无返回值。

代码示例
:::::::::
COPY-FROM: paddle.distributed.scatter_object_list
