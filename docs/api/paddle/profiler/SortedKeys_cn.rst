.. _cn_api_paddle_profiler_SortedKeys:

SortedKeys
---------------------

.. py:class:: paddle.profiler.SortedKeys


SortedKeys 枚举类用来指定打印的统计 :ref:`表单 <cn_api_paddle_profiler_Profiler_summary>` 内数据的排序方式。

排序方式说明
::::::::::::

    - **SortedKeys.CPUTotal** - 按活动的 CPU 总时间排序。
    - **SortedKeys.CPUAvg**  - 按活动的 CPU 平均时间排序。
    - **SortedKeys.CPUMax**  - 按活动的 CPU 上最大时间排序。
    - **SortedKeys.CPUMin**  - 按活动的 CPU 上最小时间排序。
    - **SortedKeys.GPUTotal**  - 按活动的 GPU 总时间排序。
    - **SortedKeys.GPUAvg**  - 按活动的 GPU 平均时间排序。
    - **SortedKeys.GPUMax**  - 按活动的 GPU 上最大时间排序。
    - **SortedKeys.GPUMin**  - 按活动的 GPU 上最小时间排序。
