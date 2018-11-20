

.. _cn_api_fluid_profiler_cuda_profiler:

fluid_profiler
>>>>>>>>>>>>

.. py:class:: paddle.fluid.profiler.cuda_profiler(*args, **kwds)

The CUDA profiler. This fuctions is used to profile CUDA program by CUDA runtime application programming interface. The profiling result will be written into output_file with Key-Value pair format or Comma separated values format. The user can set the output mode by output_mode argument and set the counters/options for profiling by config argument. The default config is [‘gpustarttimestamp’, ‘gpustarttimestamp’, ‘gridsize3d’, ‘threadblocksize’, ‘streamid’, ‘enableonstart 0’, ‘conckerneltrace’]. Then users can use NVIDIA Visual Profiler (https://developer.nvidia.com/nvidia-visual-profiler) tools to load this this output file to visualize results.

CUDA分析器。通过CUDA运行时应用程序编程接口对CUDA程序进行概要分析。分析结果将以键-值对格式或逗号分隔值格式写入output_file。用户可以通过output_mode参数设置输出模式，并通过配置参数设置计数器/选项。默认配置是[' gpustarttimestamp '， ' gpustarttimestamp '， ' gridsize3d '， ' threadblocksize '， ' streamid '， ' enableonstart 0 '， ' conckerneltrace ']。然后，用户可以使用NVIDIA Visual Profiler (https://developer.nvidia.com/nvidia-visualprofiler)工具来加载这个输出文件以可视化结果。


参数:
  - **output_file** (string) – 输出文件名称, 输出结果将会写入该文件
  - **output_mode** (string) – 输出格式是有 key-value 键值对 并用逗号割。格式应该是' kvp '或' csv '
  - **config** (list of string) – 参考“Compute Command Line Profiler User Guide” 查阅 profiler options 和 counter相关信息

**代码示例**


..  code-block:: python
  
    import paddle.fluid as fluid
    import paddle.fluid.profiler as profiler

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
    # then use  NVIDIA Visual Profiler (nvvp) to load this output file
    # to visualize results.

