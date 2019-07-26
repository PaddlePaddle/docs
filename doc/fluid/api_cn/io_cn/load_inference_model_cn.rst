.. _cn_api_fluid_io_load_inference_model:

load_inference_model
-------------------------------

.. py:function:: paddle.fluid.io.load_inference_model(dirname, executor, model_filename=None, params_filename=None, pserver_endpoints=None)

从指定目录中加载预测模型(inference model)。通过这个API，您可以获得模型结构（预测程序）和模型参数。如果您只想下载预训练后的模型的参数，请使用load_params API。更多细节请参考 ``模型/变量的保存、载入与增量训练`` 。

参数:
  - **dirname** (str) – model的路径
  - **executor** (Executor) – 运行 inference model的 ``executor``
  - **model_filename** (str|None) –  存储着预测 Program 的文件名称。如果设置为None，将使用默认的文件名为： ``__model__``
  - **params_filename** (str|None) –  加载所有相关参数的文件名称。如果设置为None，则参数将保存在单独的文件中。
  - **pserver_endpoints** (list|None) – 只有在分布式预测时需要用到。 当在训练时使用分布式 look up table , 需要这个参数. 该参数是 pserver endpoints 的列表

返回: 这个函数的返回有三个元素的元组(Program，feed_target_names, fetch_targets)。Program 是一个 ``Program`` ，它是预测 ``Program``。  ``feed_target_names`` 是一个str列表，它包含需要在预测 ``Program`` 中提供数据的变量的名称。``fetch_targets`` 是一个 ``Variable`` 列表，从中我们可以得到推断结果。

返回类型：元组(tuple)

抛出异常：
   - ``ValueError`` – 如果 ``dirname`` 非法 

.. code-block:: python

        import paddle.fluid as fluid
        import numpy as np
        main_prog = fluid.Program()
        startup_prog = fluid.Program()
        with fluid.program_guard(main_prog, startup_prog):
            data = fluid.layers.data(name="img", shape=[64, 784], append_batch_size=False)
            w = fluid.layers.create_parameter(shape=[784, 200], dtype='float32')
            b = fluid.layers.create_parameter(shape=[200], dtype='float32')
            hidden_w = fluid.layers.matmul(x=data, y=w)
            hidden_b = fluid.layers.elementwise_add(hidden_w, b)
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_prog)
        path = "./infer_model"
        fluid.io.save_inference_model(dirname=path, feeded_var_names=['img'],target_vars=[hidden_b], executor=exe, main_program=main_prog)
        tensor_img = np.array(np.random.random((1, 64, 784)), dtype=np.float32)
        [inference_program, feed_target_names, fetch_targets] = (fluid.io.load_inference_model(dirname=path, executor=exe))
        
        results = exe.run(inference_program,
                  feed={feed_target_names[0]: tensor_img},
                  fetch_list=fetch_targets)

        # endpoints是pserver服务器终端列表，下面仅为一个样例
        endpoints = ["127.0.0.1:2023","127.0.0.1:2024"]
        # 如果需要查询表格，我们可以使用：
        [dist_inference_program, dist_feed_target_names, dist_fetch_targets] = (
            fluid.io.load_inference_model(dirname=path,
                                          executor=exe,
                                          pserver_endpoints=endpoints))

        # 在这个示例中，inference program 保存在“ ./infer_model/__model__”中
        # 参数保存在“./infer_mode ”单独的若干文件中
        # 加载 inference program 后， executor 使用 fetch_targets 和 feed_target_names 执行Program，得到预测结果







