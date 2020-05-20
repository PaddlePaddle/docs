.. _cn_api_fluid_io_load_inference_model:

load_inference_model
-------------------------------


.. py:function:: paddle.fluid.io.load_inference_model(dirname, executor, model_filename=None, params_filename=None, pserver_endpoints=None)

:api_attr: 声明式编程模式（静态图)



从指定文件路径中加载预测模型(Inference Model)，即调用该接口可获得模型结构（Inference Program）和模型参数。若只想加载预训练后的模型参数，请使用 :ref:`cn_api_fluid_io_load_params` 接口。更多细节请参考 :ref:`api_guide_model_save_reader` 。

参数：
  - **dirname** (str) – 待加载模型的存储路径。
  - **executor** (Executor) – 运行 Inference Model 的 ``executor`` ，详见 :ref:`api_guide_executor` 。
  - **model_filename** (str，可选) –  存储Inference Program结构的文件名称。如果设置为None，则使用 ``__model__`` 作为默认的文件名。默认值为None。
  - **params_filename** (str，可选) –  存储所有模型参数的文件名称。当且仅当所有模型参数被保存在一个单独的二进制文件中，它才需要被指定。如果模型参数是存储在各自分离的文件中，设置它的值为None。默认值为None。
  - **pserver_endpoints** (list，可选) – 只有在分布式预测时才需要用到。当训练过程中使用分布式查找表(distributed lookup table)时, 预测时需要指定pserver_endpoints的值。它是 pserver endpoints 的列表，默认值为None。

返回：该接口返回一个包含三个元素的列表(program，feed_target_names, fetch_targets)。它们的含义描述如下：
  - **program** （Program）– ``Program`` （详见 :ref:`api_guide_Program` ）类的实例。此处它被用于预测，因此可被称为Inference Program。
  - **feed_target_names** （list）– 字符串列表，包含着Inference Program预测时所需提供数据的所有变量名称（即所有输入变量的名称）。
  - **fetch_targets** （list）– ``Variable`` （详见 :ref:`api_guide_Program` ）类型列表，包含着模型的所有输出变量。通过这些输出变量即可得到模型的预测结果。

**返回类型：** 列表（list）

抛出异常：
  - ``ValueError`` – 如果接口参数 ``dirname`` 指向一个不存在的文件路径，则抛出异常。

**代码示例**

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    import numpy as np
    
    # 构建模型
    main_prog = paddle.Program()
    startup_prog = paddle.Program()
    with paddle.program_guard(main_prog, startup_prog):
        data = fluid.layers.data(name='img', shape=[64, 784], append_batch_size
            =False)
        w = paddle.create_parameter(shape=[784, 200], dtype='float32')
        b = paddle.create_parameter(shape=[200], dtype='float32')
        hidden_w = paddle.mm(x=data, y=w, out=None)
        hidden_b = paddle.add(hidden_w, b, alpha=1, out=None)
    place = paddle.CPUPlace()
    exe = paddle.Executor(place)
    exe.run(startup_prog)
    
    # 保存预测模型
    path = './infer_model'
    paddle.io.save_inference_model(dirname=path, feeded_var_names=['img'],
        target_vars=[hidden_b], executor=exe, main_program=main_prog)
    [inference_program, feed_target_names, fetch_targets
        ] = paddle.io.load_inference_model(dirname=path, executor=exe)
    tensor_img = np.array(np.random.random((1, 64, 784)), dtype=np.float32)
    results = exe.run(inference_program, feed={feed_target_names[0]: tensor_img
        }, fetch_list=fetch_targets)
    endpoints = ['127.0.0.1:2023', '127.0.0.1:2024']
    [dist_inference_program, dist_feed_target_names, dist_fetch_targets
        ] = paddle.io.load_inference_model(dirname=path, executor=exe,
        pserver_endpoints=endpoints)

