.. _cn_api_paddle_distributed_broadcast_object_list:

broadcast_object_list
-------------------------------


.. py:function:: paddle.distributed.broadcast_object_list(object_list, src, group=None)

将一个 picklable 对象发送到每个进程。
过程与 ``broadcast`` 类似，但可以传入自定义的 python 对象。

.. note::
  该 API 只支持动态图模式。

参数
:::::::::
    - **object_list** (List[Any]) - 在目标进程上为待广播的 object 列表，在其他进程上为用于接收广播结果的 object 列表。
    - **src** (int) - 目标进程的 rank，该进程传入的 object 列表将被发送到其他进程上。
    - **group** (Group，可选) - 执行该操作的进程组实例（通过 ``new_group`` 创建）。默认为 None，即使用全局默认进程组。

返回
:::::::::
无返回值。

代码示例
:::::::::
COPY-FROM: paddle.distributed.broadcast_object_list
