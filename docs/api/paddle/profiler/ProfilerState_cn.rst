.. _cn_api_profiler_profilerstate:

ProfilerState
---------------------

.. py:class:: paddle.profiler.ProfilerState


ProfilerState 枚举类用来表示 :ref:`性能分析器 <cn_api_profiler_profiler>` 的状态。

状态说明
::::::::::::

    - **ProfilerState.CLOSED** - 性能分析器处于"关闭"状态，不收集任何性能数据。
    - **ProfilerState.READY**  - 性能分析器处于"准备"状态，性能分析器开启，但是不做数据记录，该状态主要为了减少性能分析器初始化时的开销对所收集的性能数据的影响。
    - **ProfilerState.RECORD** - 性能分析器处于"记录"状态，性能分析器正常工作，并且记录性能数据。
    - **ProfilerState.RECORD_AND_RETURN** - 性能分析器处于"记录并返回"状态，性能分析器正常工作，该状态可视为一个性能分析周期中最后一个"RECORD"状态，并且会将记录的性能数据返回。
