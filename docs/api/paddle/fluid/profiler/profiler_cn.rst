.. _cn_api_fluid_profiler_profiler:

profiler
-------------------------------

.. py:function:: paddle.fluid.profiler.profiler(state, sorted_key=None, profile_path='/tmp/profile', tracer_option='Default')

通用性能分析器。与 :ref:`cn_api_fluid_profiler_cuda_profiler` 不同，此分析器可用于分析CPU和GPU程序。

.. warning::
   该API将在未来废弃，对CPU和GPU的性能分析请参考使用paddle最新的性能分析器 :ref:`Profiler <cn_api_profiler_profiler>` 。
   如代码示例中对该API的使用用新的Profiler进行替换最简单的用法如下

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    import paddle.profiler as profiler
    import numpy as np

    paddle.enable_static()
    epoc = 8
    dshape = [4, 3, 28, 28]
    data = fluid.layers.data(name='data', shape=[3, 28, 28], dtype='float32')
    conv = fluid.layers.conv2d(data, 20, 3, stride=[1, 1], padding=[1, 1])

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    with profiler.Profiler() as prof:
        for i in range(epoc):
            input = np.random.random(dshape).astype('float32')
            exe.run(fluid.default_main_program(), feed={'data': input})
            prof.step()
    prof.summary()

参数
::::::::::::

  - **state** (str) –  性能分析状态，取值为 'CPU' 或 'GPU' 或 'All'。'CPU'表示只分析CPU上的性能；'GPU'表示同时分析CPU和GPU上的性能；'All'表示除了同时分析CPU和GPU上的性能外，还将生成 `性能分析的时间轴信息 <../../advanced_usage/development/profiling/timeline_cn.html>`_ 。
  - **sorted_key** (str，可选) – 性能分析结果的打印顺序，取值为None、'call'、'total'、'max'、'min'、'ave'之一。默认值为None，表示按照第一次结束时间顺序打印；'call'表示按调用的数量进行排序；'total'表示按总执行时间排序；'max'表示按最大执行时间排序；'min'表示按最小执行时间排序；'ave'表示按平均执行时间排序。
  - **profile_path** (str，可选) –  如果性能分析状态为'All'，将生成的时间轴信息写入profile_path，默认输出文件为 ``/tmp/profile`` 。
  - **tracer_option** (str，可选) –   性能分析选项取值为 'Default' 或 'OpDetail' 或 'AllOpDetail'，此选项用于设置性能分析层次并打印不同层次的性能分析结果，`Default` 选项打印不同Op类型的性能分析结果，`OpDetail` 则会打印不同OP类型更详细的性能分析结果，比如compute和data transform。 `AllOpDetail` 和 `OpDetail` 类似，但是打印的是不同Op名字的性能分析结果。

抛出异常
::::::::::::

  - ``ValueError`` – 如果state取值不在 ['CPU', 'GPU', 'All']中，或sorted_key取值不在 [None, 'calls', 'total', 'max', 'min', 'ave']中，则抛出异常。

代码示例
::::::::::::

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    import paddle.fluid.profiler as profiler
    import numpy as np

    paddle.enable_static()
    epoc = 8
    dshape = [4, 3, 28, 28]
    data = fluid.layers.data(name='data', shape=[3, 28, 28], dtype='float32')
    conv = fluid.layers.conv2d(data, 20, 3, stride=[1, 1], padding=[1, 1])

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    with profiler.profiler('CPU', 'total', '/tmp/profile') as prof:
        for i in range(epoc):
            input = np.random.random(dshape).astype('float32')
            exe.run(fluid.default_main_program(), feed={'data': input})

**结果示例**

.. code-block:: text

    #### sorted_key = 'total', 'calls', 'max', 'min', 'ave' 结果 ####
    # 示例结果中，除了Sorted by number of xxx in descending order in the same thread 这句随着sorted_key变化而不同，其余均相同。
    # 原因是，示例结果中，上述5列都已经按从大到小排列了。
    ------------------------->     Profiling Report     <-------------------------

    Place: CPU
    Time unit: ms
    Sorted by total time in descending order in the same thread
    #Sorted by number of calls in descending order in the same thread
    #Sorted by number of max in descending order in the same thread
    #Sorted by number of min in descending order in the same thread
    #Sorted by number of avg in descending order in the same thread

    Event                       Calls       Total       Min.        Max.        Ave.        Ratio.
    thread0::conv2d             8           129.406     0.304303    127.076     16.1758     0.983319
    thread0::elementwise_add    8           2.11865     0.193486    0.525592    0.264832    0.016099
    thread0::feed               8           0.076649    0.006834    0.024616    0.00958112  0.000582432

    #### sorted_key = None 结果 ####
    # 示例结果中，是按照Op结束时间顺序打印，因此打印顺序为feed->conv2d->elementwise_add
    ------------------------->     Profiling Report     <-------------------------

    Place: CPU
    Time unit: ms
    Sorted by event first end time in descending order in the same thread

    Event                       Calls       Total       Min.        Max.        Ave.        Ratio.
    thread0::feed               8           0.077419    0.006608    0.023349    0.00967738  0.00775934
    thread0::conv2d             8           7.93456     0.291385    5.63342     0.99182     0.795243
    thread0::elementwise_add    8           1.96555     0.191884    0.518004    0.245693    0.196998
