.. _cn_api_fluid_profiler_start_profiler:

start_profiler
-------------------------------

.. py:function:: paddle.fluid.profiler.start_profiler(state)

激活使用性能分析器。除了 :ref:`cn_api_fluid_profiler_profiler` 外，用户还可以使用 :ref:`cn_api_fluid_profiler_start_profiler` 和 :ref:`cn_api_fluid_profiler_stop_profiler` 来激活和停止使用性能分析器。

参数:
  - **state** (str) –  性能分析状态, 取值为 'CPU' 或 'GPU' 或 'All'。'CPU'表示只分析CPU上的性能；'GPU'表示同时分析CPU和GPU上的性能；'All'表示除了同时分析CPU和GPU上的性能外，还将生成性能分析的时间轴信息 :ref:`fluid_timeline` 。

抛出异常:
  - ``ValueError`` – 如果state取值不在 ['CPU', 'GPU', 'All']中，则抛出异常。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    import paddle.fluid.profiler as profiler

    profiler.start_profiler('GPU')
    for iter in range(10):
        if iter == 2:
            profiler.reset_profiler()
        # except each iteration
    profiler.stop_profiler('total', '/tmp/profile')
