.. _cn_api_profiler_profilertarget:

ProfilerTarget
---------------------

.. py:class:: paddle.profiler.ProfilerTarget


ProfilerTarget 枚举类用来指定 :ref:`性能分析 <cn_api_profiler_profiler>` 的设备。目前仅支持 CPU，GPU 和 MLU。

设备说明
::::::::::::

    - **ProfilerTarget.CPU** - 性能分析对象的设备为CPU。
    - **ProfilerTarget.GPU**  - 性能分析对象的设备为GPU。
    - **ProfilerTarget.MLU**  - 性能分析对象的设备为MLU。
