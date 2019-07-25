.. _cn_api_fluid_profiler_start_profiler:

start_profiler
-------------------------------

.. py:function:: paddle.fluid.profiler.start_profiler(state)

激活使用 profiler， 用户可以使用 ``fluid.profiler.start_profiler`` 和 ``fluid.profiler.stop_profiler`` 插入代码
不能使用 ``fluid.profiler.profiler``


如果 state== ' All '，在profile_path 中写入文件 profile proto 。该文件记录执行期间的时间顺序信息。然后用户可以看到这个文件的时间轴，请参考 `这里 <../advanced_usage/development/profiling/timeline_cn.html>`_

参数:
  - **state** (string) – profiling state, 取值为 'CPU' 或 'GPU' 或 'All', 'CPU' 代表只分析 cpu. 'GPU' 代表只分析 GPU . 'All' 会产生 timeline.

抛出异常:
  - ``ValueError`` – 如果state 取值不在 ['CPU', 'GPU', 'All']中

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

                # ...








