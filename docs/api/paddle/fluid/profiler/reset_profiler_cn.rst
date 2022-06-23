.. _cn_api_fluid_profiler_reset_profiler:

reset_profiler
-------------------------------

.. py:function:: paddle.fluid.profiler.reset_profiler()

清除之前的性能分析记录。此接口不能和 :ref:`cn_api_fluid_profiler_cuda_profiler` 一起使用，但它可以和 :ref:`cn_api_fluid_profiler_start_profiler` 、:ref:`cn_api_fluid_profiler_stop_profiler` 和 :ref:`cn_api_fluid_profiler_profiler` 一起使用。

.. warning::
   该API将在未来废弃，对CPU和GPU的性能分析请参考使用paddle最新的性能分析器 :ref:`Profiler <cn_api_profiler_profiler>` 。

代码示例
::::::::::::

COPY-FROM: paddle.fluid.profiler.reset_profiler