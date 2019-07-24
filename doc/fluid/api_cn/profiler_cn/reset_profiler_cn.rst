.. _cn_api_fluid_profiler_reset_profiler:

reset_profiler
-------------------------------

.. py:function:: paddle.fluid.profiler.reset_profiler()

清除之前的时间记录。此接口不适用于 ``fluid.profiler.cuda_profiler`` ，它只适用于 ``fluid.profiler.start_profiler`` , ``fluid.profiler.stop_profiler`` , ``fluid.profiler.profiler`` 。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    import paddle.fluid.profiler as profiler
    with profiler.profiler('CPU', 'total', '/tmp/profile'):
    for iter in range(10):
        if iter == 2:
            profiler.reset_profiler()
        # ...








