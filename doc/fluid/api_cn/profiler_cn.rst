#################
 fluid.profiler
#################



.. _cn_api_fluid_profiler_cuda_profiler:

cuda_profiler
-------------------------------

.. py:function:: paddle.fluid.profiler.cuda_profiler(output_file, output_mode=None, config=None)


CUDA分析器。通过CUDA运行时应用程序编程接口对CUDA程序进行性能分析。分析结果将以键-值对格式或逗号分隔的格式写入output_file。用户可以通过output_mode参数设置输出模式，并通过配置参数设置计数器/选项。默认配置是[' gpustarttimestamp '， ' gpustarttimestamp '， ' gridsize3d '， ' threadblocksize '， ' streamid '， ' enableonstart 0 '， ' conckerneltrace ']。然后，用户可使用 `NVIDIA Visual Profiler <https://developer.nvidia.com/nvidia-visual-profiler>`_ 工具来加载这个输出文件以可视化结果。


参数:
  - **output_file** (string) – 输出文件名称, 输出结果将会写入该文件
  - **output_mode** (string) – 输出格式是有 key-value 键值对 和 逗号的分割的格式。格式应该是' kvp '或' csv '
  - **config** (list of string) – 参考"Compute Command Line Profiler User Guide" 查阅 profiler options 和 counter相关信息

抛出异常:
    - ``ValueError`` -  如果 ``output_mode`` 不在 ['kvp', 'csv'] 中


**代码示例**


.. code-block:: python

    import paddle.fluid as fluid
    import paddle.fluid.profiler as profiler
    import numpy as np

    epoc = 8
    dshape = [4, 3, 28, 28]
    data = fluid.layers.data(name='data', shape=[3, 28, 28], dtype='float32')
    conv = fluid.layers.conv2d(data, 20, 3, stride=[1, 1], padding=[1, 1])

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    output_file = 'cuda_profiler.txt'
    with profiler.cuda_profiler(output_file, 'csv') as nvprof:
        for i in range(epoc):
            input = np.random.random(dshape).astype('float32')
            exe.run(fluid.default_main_program(), feed={'data': input})

    # 之后可以使用 NVIDIA Visual Profile 可视化结果









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







.. _cn_api_fluid_profiler_reset_profiler:

reset_profiler
-------------------------------

.. py:function:: paddle.fluid.profiler.reset_profiler()

清除之前的时间记录。此接口不适用于 ``fluid.profiler.cuda_profiler`` ，它只适用于 ``fluid.profiler.start_profiler`` , ``fluid.profiler.stop_profiler`` , ``fluid.profiler.profiler`` 。

**代码示例**

.. code-block:: python

    import paddle.fluid.profiler as profiler
    with profiler.profiler('CPU', 'total', '/tmp/profile'):
    for iter in range(10):
        if iter == 2:
            profiler.reset_profiler()
        # ...








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

    import paddle.fluid.profiler as profiler

    profiler.start_profiler('GPU')
    for iter in range(10):
        if iter == 2:
            profiler.reset_profiler()
        # except each iteration
    profiler.stop_profiler('total', '/tmp/profile')

                # ...








.. _cn_api_fluid_profiler_stop_profiler:

stop_profiler
-------------------------------

.. py:function:: paddle.fluid.profiler.stop_profiler(sorted_key=None, profile_path='/tmp/profile')

停止 profiler， 用户可以使用 ``fluid.profiler.start_profiler`` 和 ``fluid.profiler.stop_profiler`` 插入代码
不能使用 ``fluid.profiler.profiler``

参数:
  - **sorted_key** (string) – 如果为None，prfile的结果将按照事件的第一次结束时间顺序打印。否则，结果将按标志排序。标志取值为"call"、"total"、"max"、"min" "ave"之一，根据调用着的数量进行排序。total表示按总执行时间排序，max 表示按最大执行时间排序。min 表示按最小执行时间排序。ave表示按平均执行时间排序。
  - **profile_path** (string) - 如果 state == 'All', 结果将写入文件 profile proto.


抛出异常:
  - ``ValueError`` – 如果state 取值不在 ['CPU', 'GPU', 'All']中

**代码示例**

.. code-block:: python

    import paddle.fluid.profiler as profiler

    profiler.start_profiler('GPU')
    for iter in range(10):
        if iter == 2:
            profiler.reset_profiler()
            # except each iteration
    profiler.stop_profiler('total', '/tmp/profile')







