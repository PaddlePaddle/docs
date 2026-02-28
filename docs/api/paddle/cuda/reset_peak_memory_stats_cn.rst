.. _cn_api_paddle_cuda_reset_peak_memory_stats:

reset_peak_memory_stats
-----------------------

.. py:function:: paddle.cuda.reset_peak_memory_stats(device=None)

重置所有设备的峰值内存统计信息。
此方法重置程序执行期间为每个设备记录的峰值内存使用情况。
它将所有设备的峰值内存使用率设置为零。
与 :ref:`cn_api_paddle_device_reset_peak_memory_stats` 功能一致

参数
::::::::::::
    - **device** (int|paddle.Place|str|None) - 设备、设备的 id 或设备的字符串名称，如 ``npu:x``，从中获取设备的属性。如果设备为 None，则该设备为当前设备，默认值：None。

代码示例
::::::::::::
COPY-FROM: paddle.cuda.reset_peak_memory_stats
