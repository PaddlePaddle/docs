.. _cn_api_profiler_profilertarget:

ProfilerTarget
---------------------

.. py:class:: paddle.profiler.ProfilerTarget


ProfilerTarget 枚举类用来指定 :ref:`性能分析 <cn_api_profiler_profiler>` 的设备。目前仅支持 CPU，GPU 和 MLU。

设备说明
::::::::::::

    - **ProfilerTarget.CPU** - 性能分析对象为 CPU 上的活动。
    - **ProfilerTarget.GPU**  - 性能分析对象为 GPU 上的活动。
    - **ProfilerTarget.MLU**  - 性能分析对象为 MLU 上的活动。
