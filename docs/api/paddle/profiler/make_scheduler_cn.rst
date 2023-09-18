.. _cn_api_paddle_profiler_make_scheduler:

make_scheduler
---------------------

.. py:function:: paddle.profiler.make_scheduler(*, closed: int, ready: int, record: int, repeat: int=0, skip_first: int=0)

生成性能分析器状态(详情见 :ref:`状态说明 <cn_api_paddle_profiler_ProfilerState>` )的调度器函数，可根据设置的参数来调度性能分析器的状态。
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

    - **closed** (int) - 处于 ProfilerState.CLOSED 状态的 step 数量。
    - **ready** (int) - 处于 ProfilerState.READY 状态的 step 数量。
    - **record** (int) - 处于 ProfilerState.RECORD 状态的 step 数量，record 的最后一个 step 会处于 ProfilerState.RECORD_AND_RETURN 状态。
    - **repeat** (int，可选) - 调度器重复该状态调度过程的次数，默认值为 0，意味着一直重复该调度过程直到性能分析器结束。
    - **skip_first** (int，可选) - 跳过前 skip_first 个 step，不参与状态调度，并处于 ProfilerState.CLOSED 状态，默认值为 0。

返回
:::::::::

调度函数（callable)，该函数会接收一个参数 step_num，并计算返回相应的 ProfilerState。调度函数会根据上述状态转换过程进行调度。


代码示例 1
::::::::::

性能分析 batch [2, 5]。

设定第 0 个 batch 处于 CLOSED，第 1 个 batch 处于 READY，第[2 - 5]个 batch 处于 RECORD，在第 5 个 batch 返回收集的性能数据。

COPY-FROM: paddle.profiler.make_scheduler:code-example1

代码示例 2
::::::::::

性能分析 batch [3,6], [9,12], [15, 18]..。

设定第 0 个 batch 跳过，第 1 个 batch 处于 CLOSED，第 2 个 batch 处于 READ，第[3 - 6]个 batch 处于 RECORD，在第 6 个 batch 返回收集的性能数据。即第 7 个 batch 处于 CLOSED，第 8 个 batch 处于 READY,
第[9-12]个 batch 处于 RECORD，并在第 12 个 batch 返回第二轮所收集到的性能数据。以此类推，直到性能分析器结束。

COPY-FROM: paddle.profiler.make_scheduler:code-example2
