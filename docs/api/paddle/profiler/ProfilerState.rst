.. _cn_api_profiler_profilerstate:

ProfilerState
---------------------

.. py:class:: paddle.profiler.ProfilerState


枚举类，用来表示性能分析器的状态。

状态说明如下：
    - **CLOSED** - "关闭"，不收集任何性能数据。
    - **READY**  - "准备"，性能分析器开启，但是不做数据记录，该状态主要为了减少性能分析器初始化时的开销对所收集的性能数据的影响。
    - **RECORD** - "记录"，性能分析器正常工作，并且记录性能数据。
    - **RECORD_AND_RETURN** - "记录并返回"，性能分析器正常工作，并且将记录的性能数据返回。
