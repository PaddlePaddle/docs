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









