.. _cn_api_fluid_profiler_reset_profiler:

reset_profiler
-------------------------------

.. py:function:: paddle.fluid.profiler.reset_profiler()




清除之前的性能分析记录。此接口不能和 :ref:`cn_api_fluid_profiler_cuda_profiler` 一起使用 ，但它可以和 :ref:`cn_api_fluid_profiler_start_profiler` 、:ref:`cn_api_fluid_profiler_stop_profiler` 和 :ref:`cn_api_fluid_profiler_profiler` 一起使用。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    import paddle.fluid.profiler as profiler
    with profiler.profiler('CPU', 'total', '/tmp/profile'):
        for iter in range(10):
            if iter == 2:
                profiler.reset_profiler()
            # ...
