.. _cn_api_fluid_profiler_profiler:

profiler
-------------------------------

.. py:function:: paddle.fluid.profiler.profiler(state, sorted_key=None, profile_path='/tmp/profile')

profile interface 。与cuda_profiler不同，此profiler可用于分析CPU和GPU程序。默认情况下，它记录CPU和GPU kernel，如果想分析其他程序，可以参考教程来在c++代码中添加更多代码。


如果 state== ' All '，在profile_path 中写入文件 profile proto 。该文件记录执行期间的时间顺序信息。然后用户可以看到这个文件的时间轴，请参考 `这里 <../advanced_usage/development/profiling/timeline_cn.html>`_

参数:
  - **state** (string) –  profiling state, 取值为 'CPU' 或 'GPU',  profiler 使用 CPU timer 或GPU timer 进行 profiling. 虽然用户可能在开始时指定了执行位置(CPUPlace/CUDAPlace)，但是为了灵活性，profiler不会使用这个位置。
  - **sorted_key** (string) – 如果为None，prfile的结果将按照事件的第一次结束时间顺序打印。否则，结果将按标志排序。标志取值为"call"、"total"、"max"、"min" "ave"之一，根据调用着的数量进行排序。total表示按总执行时间排序，max 表示按最大执行时间排序。min 表示按最小执行时间排序。ave表示按平均执行时间排序。
  - **profile_path** (string) –  如果 state == 'All', 结果将写入文件 profile proto.

抛出异常：
  - ``ValueError`` – 如果state 取值不在 ['CPU', 'GPU', 'All']中. 如果 sorted_key 取值不在 ['calls', 'total', 'max', 'min', 'ave']

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    import paddle.fluid.profiler as profiler
    import numpy as np

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







