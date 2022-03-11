.. _cn_api_profiler_profilertarget:

ProfilerTarget
---------------------

.. py:class:: paddle.profiler.ProfilerTarget


枚举类，用来指定性能分析的设备。目前仅支持CPU和GPU。

设备说明如下：
    - **ProfilerTarget.CPU** - 性能分析对象为CPU上的活动
    - **ProfilerTarget.GPU**  - 性能分析对象为GPU上的活动