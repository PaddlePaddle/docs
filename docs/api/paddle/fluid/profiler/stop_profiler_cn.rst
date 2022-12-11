.. _cn_api_fluid_profiler_stop_profiler:

stop_profiler
-------------------------------

.. py:function:: paddle.fluid.profiler.stop_profiler(sorted_key=None, profile_path='/tmp/profile')




停止使用性能分析器。除了 :ref:`cn_api_fluid_profiler_profiler` 外，用户还可以使用 :ref:`cn_api_fluid_profiler_start_profiler` 和 :ref:`cn_api_fluid_profiler_stop_profiler` 来激活和停止使用性能分析器。

.. warning::
  该API将在未来废弃，对CPU和GPU的性能分析请参考使用paddle最新的性能分析器 :ref:`Profiler <cn_api_profiler_profiler>`。
  对于关闭profiler，使用新的接口来替换该接口的使用有下列两种方式

.. code-block:: python

  #使用新的接口替换该接口的使用方式
  #1。调用Profiler对象的stop接口
  import paddle
  import paddle.profiler as profiler
  prof = profiler.Profiler()
  prof.start()
  for iter in range(10):
      #train()
      prof.step()
  prof.stop()
  prof.summary() # 打印表单

.. code-block:: python

  #2。使用环境管理器的用法
  import paddle
  import paddle.profiler as profiler
  with profiler.Profiler() as prof:
    for iter in range(10):
      #train()
      prof.step()
  prof.summary() # 打印表单

  

参数
::::::::::::

  - **sorted_key** (str，可选) – 性能分析结果的打印顺序，取值为None、'call'、'total'、'max'、'min'、'ave'之一。默认值为None，表示按照第一次结束时间顺序打印；'call'表示按调用的数量进行排序；'total'表示按总执行时间排序；'max'表示按最大执行时间排序；'min'表示按最小执行时间排序；'ave'表示按平均执行时间排序。
  - **profile_path** (str，可选) –  如果性能分析状态为'All'，将生成的时间轴信息写入profile_path，默认输出文件为 ``/tmp/profile`` 。


抛出异常
::::::::::::

  - ``ValueError`` – 如果sorted_key取值不在 [None, 'calls', 'total', 'max', 'min', 'ave']中，则抛出异常。

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
