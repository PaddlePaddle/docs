.. _cn_api_profiler_make_scheduler:

make_scheduler
---------------------

.. py:function:: paddle.profiler.make_scheduler(*, closed: int, ready: int, record: int, repeat: int=0, skip_first: int=0)

该接口用于生成性能分析器状态(详情见:ref:`状态说明 <cn_api_profiler_profilerstate>`)的调度器。

参数:
    - **closed** (int) - 处于ProfilerState.CLOSED状态的step数量。
    - **ready** (int) - 处于ProfilerState.CLOSED状态的step数量。
    - **record** (int) - 处于ProfilerState.RECORD状态的step数量, record的最后一个step会处于ProfilerState.RECORD_AND_RETURN状态。
    - **repeat** (int, 可选) - 调度器重复该状态调度过程的次数, 默认值为0意味着一直重复该调度过程。
    - **skip_first** (int, 可选) - 跳过前skip_first个step，不参与状态调度，并处于ProfilerState.CLOSED状态。

返回: 调度函数（callable), 该函数会接收一个参数step_num，并计算返回相应的ProfilerState。性能分析器可以通过该调度函数更新状态。


**代码示例**

1. 第0个step处于CLOSED， 第[1 - 2]个step处于READY, 第[3 - 5]个step处于RECORD，在第5个step返回收集的性能数据。

.. code-block:: python

    import paddle.profiler as profiler
    sheduler = profiler.make_scheduler(closed=1, ready=2, record=3, repeat=1)

2. 第0个step处于CLOSED， 第[1 - 3]个step处于READY, 第[4 - 6]个step处于RECORD，在第5个step返回收集的性能数据，重复3次。即第7个step处于CLOSED，第[8-10]个step处于READY,
第[11-13]个step处于RECORD，并在第13个step返回第二轮所收集到的性能数据。以此类推，直到第20个step返回第三轮所收集到的性能数据，调度结束。

.. code-block:: python

    import paddle.profiler as profiler
    sheduler = profiler.make_scheduler(closed=1, ready=3, record=3, repeat=3)
