.. _cn_api_paddle_distributed_all_gather_object:

all_gather_object
-------------------------------


.. py:function:: paddle.distributed.all_gather_object(object_list, obj, group=None)

组聚合，聚合进程组内指定的 picklable 对象，随后将聚合后的对象列表发送到每个进程。
过程与 ``all_gather`` 类似，但可以传入自定义的 python 对象。

.. note::
  该 API 只支持动态图模式。

参数
:::::::::
    - **object_list** (List[Any]) - 用于保存聚合结果的列表。
    - **object** (Any) - 待聚合的对象。需要保证该对象是 picklable 的。
    - **group** (Group，可选) - 执行该操作的进程组实例（通过 ``new_group`` 创建）。默认为 None，即使用全局默认进程组。

返回
:::::::::
无返回值。

代码示例
:::::::::
COPY-FROM: paddle.distributed.all_gather_object
