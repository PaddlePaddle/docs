.. _cn_api_fluid_profiler_cuda_profiler:

cuda_profiler
-------------------------------

.. py:function:: paddle.fluid.profiler.cuda_profiler(output_file, output_mode=None, config=None)





CUDA性能分析器。该分析器通过调用CUDA运行时编程接口，对CUDA程序进行性能分析，并将分析结果写入输出文件output_file。输出格式由output_mode参数控制，性能分析配置选项由config参数控制。得到输出文件后，用户可使用 `NVIDIA Visual Profiler <https://developer.nvidia.com/nvidia-visual-profiler>`_ 工具来加载这个输出文件以获得可视化结果。


参数:
  - **output_file** (str) – 输出文件名称, 输出结果将会写入该文件。
  - **output_mode** (str，可选) – 输出格式，有两种可以选择，分别是 key-value 键值对格式'kvp' 和 逗号分割的格式'csv'（默认格式）。
  - **config** (list<str>, 可选) – NVIDIA性能分析配置列表，默认值为None时会选择以下配置：['gpustarttimestamp', 'gpuendtimestamp', 'gridsize3d', 'threadblocksize', 'streamid', 'enableonstart 0', 'conckerneltrace']。上述每个配置的含义和更多配置选项，请参考 `Compute Command Line Profiler User Guide <https://developer.download.nvidia.cn/compute/DevZone/docs/html/C/doc/Compute_Command_Line_Profiler_User_Guide.pdf>`_ 。

抛出异常:
    - ``ValueError`` -  如果输出格式output_mode不是'kvp'、'csv'两者之一，会抛出异常。

返回: 无

**代码示例**


.. code-block:: python

    import paddle.fluid as fluid
    import paddle.fluid.profiler as profiler
    import numpy as np

    epoc = 8
    dshape = [4, 3, 28, 28]
    data = fluid.data(name='data', shape=[None, 3, 28, 28], dtype='float32')
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
