.. _cn_api_fluid_device_guard:

device_guard
-------------------------------

**注意：该API仅支持【静态图】模式**

.. py:function:: paddle.fluid.device_guard(device=None)

一个用于指定OP运行设备的上下文管理器。

参数：
    - **device** (str|None) – 指定上下文中使用的设备。它可以是'cpu'或者'gpu‘，当它被设置为'cpu'或者'gpu'时，创建在该上下文中的OP将被运行在CPUPlace或者CUDAPlace上。若设置为'gpu'，同时程序运行在单卡模式下，设备的索引将与执行器的设备索引保持一致。默认值：None，在该上下文中的OP将被自动地分配设备。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    support_gpu = fluid.is_compiled_with_cuda()
    place = fluid.CPUPlace()
    if support_gpu:
        place = fluid.CUDAPlace(0)
    # if GPU is supported, the three OPs below will be automatically assigned to CUDAPlace(0)
    data1 = fluid.layers.fill_constant(shape=[1, 3, 8, 8], value=0.5, dtype='float32')
    data2 = fluid.layers.fill_constant(shape=[1, 3, 5, 5], value=0.5, dtype='float32')
    shape = fluid.layers.shape(data2)
    with fluid.device_guard("cpu"):
        # Ops created here will be placed on CPUPlace
        shape = fluid.layers.slice(shape, axes=[0], starts=[0], ends=[4])
    with fluid.device_guard('gpu'):
        # if GPU is supported, OPs created here will be placed on CUDAPlace(0), otherwise on CPUPlace
        out = fluid.layers.crop_tensor(data1, shape=shape)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    result = exe.run(fetch_list=[out])
