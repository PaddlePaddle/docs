.. _cn_api_fluid_profiler_start_profiler:

start_profiler
-------------------------------

.. py:function:: paddle.fluid.profiler.start_profiler(state, tracer_option='Default')




激活使用性能分析器。除了 :ref:`cn_api_fluid_profiler_profiler` 外，用户还可以使用 :ref:`cn_api_fluid_profiler_start_profiler` 和 :ref:`cn_api_fluid_profiler_stop_profiler` 来激活和停止使用性能分析器。

.. warning::
  该API将在未来废弃，对CPU和GPU的性能分析请参考使用paddle最新的性能分析器 :ref:`Profiler <cn_api_profiler_profiler>`。
  对于开启profiler，使用新的接口来替换该接口的使用有下列两种方式

.. code-block:: python

  #使用新的接口替换该接口的使用方式
  #1。创建Profiler对象，并调用start接口
  import paddle
  import paddle.profiler as profiler
  prof = profiler.Profiler()
  prof.start()
  for iter in range(10):
      #train()
      prof.step()
  prof.stop()

.. code-block:: python

  #2。使用环境管理器的用法
  import paddle
  import paddle.profiler as profiler
  with profiler.Profiler() as prof:
    for iter in range(10):
      #train()
      prof.step()

参数
::::::::::::

  - **state** (str) –  性能分析状态，取值为 'CPU' 或 'GPU' 或 'All'。'CPU'表示只分析CPU上的性能；'GPU'表示同时分析CPU和GPU上的性能；'All'表示除了同时分析CPU和GPU上的性能外，还将生成性能分析的时间轴信息 :ref:`fluid_timeline` 。
  - **tracer_option** (str，可选) –   性能分析选项取值为 'Default' 或 'OpDetail' 或 'AllOpDetail'，此选项用于设置性能分析层次并打印不同层次的性能分析结果，`Default` 选项打印不同Op类型的性能分析结果，`OpDetail` 则会打印不同OP类型更详细的性能分析结果，比如compute和data transform。 `AllOpDetail` 和 `OpDetail` 类似，但是打印的是不同Op名字的性能分析结果。

抛出异常
::::::::::::

  - ``ValueError`` – 如果state取值不在 ['CPU', 'GPU', 'All']中或者tracer_option取值不在['Default', 'OpDetail', 'AllOpDetail']中，则抛出异常

代码示例
::::::::::::

.. code-block:: python

    import paddle.fluid as fluid
    import paddle.fluid.profiler as profiler

    profiler.start_profiler('GPU')
    for iter in range(10):
        if iter == 2:
            profiler.reset_profiler()
        # except each iteration
    profiler.stop_profiler('total', '/tmp/profile')
