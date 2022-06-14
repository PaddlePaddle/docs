.. _cn_api_profiler_make_scheduler:

make_scheduler
---------------------

.. py:function:: paddle.profiler.make_scheduler(*, closed: int, ready: int, record: int, repeat: int=0, skip_first: int=0)

该接口用于生成性能分析器状态(详情见 :ref:`状态说明 <cn_api_profiler_profilerstate>` )的调度器函数，可根据设置的参数来调度性能分析器的状态。
调度器用于调度如下状态转换过程：

.. code-block:: text

        (CLOSED)  (CLOSED)    (CLOSED)  (READY)    (RECORD,last RETURN)      (CLOSED)
        START -> skip_first -> closed -> ready    ->    record       ->      END
                                |                        |
                                |                        | (if has_repeated < repeat)
                                - - - - - - - - - - - -

        注：repeat <= 0 意味着该状态转换过程会持续到性能分析器结束。

参数
:::::::::

    - **closed** (int) - 处于ProfilerState.CLOSED状态的step数量。
    - **ready** (int) - 处于ProfilerState.CLOSED状态的step数量。
    - **record** (int) - 处于ProfilerState.RECORD状态的step数，record的最后一个step会处于ProfilerState.RECORD_AND_RETURN状态。
    - **repeat** (int，可选) - 调度器重复该状态调度过程的次，默认值为0，味着一直重复该调度过程直到性能分析器结束。
    - **skip_first** (int，可选) - 跳过前skip_first个step，不参与状态调度，并处于ProfilerState.CLOSED状态，默认值为0。

返回
:::::::::

调度函数（callable)，该函数会接收一个参数step_num，并计算返回相应的ProfilerState。调度函数会根据上述状态转换过程进行调度。


代码示例 1
::::::::::

性能分析 batch [2, 5]。

设定第0个batch处于CLOSED，第1个batch处于READY，第[2 - 5]个batch处于RECORD，在第5个batch返回收集的性能数据。

COPY-FROM: paddle.profiler.make_scheduler:code-example1

代码示例 2
::::::::::

性能分析 batch [3,6], [9,12], [15, 18]... 

设定第0个batch跳过，第1个batch处于CLOSED，第2个batch处于READ，第[3 - 6]个batch处于RECORD，在第6个batch返回收集的性能数据。即第7个batch处于CLOSED，第8个batch处于READY,
第[9-12]个batch处于RECORD，并在第12个batch返回第二轮所收集到的性能数据。以此类推，直到性能分析器结束。

COPY-FROM: paddle.profiler.make_scheduler:code-example2
