.. _cn_api_profiler_sortedkeys:

SortedKeys
---------------------

.. py:class:: paddle.profiler.SortedKeys


SortedKeys枚举类用来指定打印的统计  :ref:`表单 <cn_api_profiler_profiler_summary>` 内数据的排序方式。

排序方式说明
::::::::::::

    - **SortedKeys.CPUTotal** - 按活动的CPU总时间排序。
    - **SortedKeys.CPUAvg**  - 按活动的CPU平均时间排序。
    - **SortedKeys.CPUMax**  - 按活动的CPU上最大时间排序。
    - **SortedKeys.CPUMin**  - 按活动的CPU上最小时间排序。
    - **SortedKeys.GPUTotal**  - 按活动的GPU总时间排序。
    - **SortedKeys.GPUAvg**  - 按活动的GPU平均时间排序。
    - **SortedKeys.GPUMax**  - 按活动的GPU上最大时间排序。
    - **SortedKeys.GPUMin**  - 按活动的GPU上最小时间排序。


