.. _cn_api_fluid_io_load_inference_model:

load_inference_model
-------------------------------


.. py:function:: paddle.static.load_inference_model(path_prefix, executor, **kwargs)




从指定文件路径中加载预测模型，包括模型结构和模型参数。

参数：
  - **path_prefix** (str) – 模型的存储目录 + 模型名称（不包含后缀）。如果是 None，表示从内存加载模型。
  - **executor** (Executor) – 运行模型的 ``executor`` ，详见 :ref:`api_guide_executor` 。
  - **kwargs** - 支持的 key 包括 'model_filename', 'params_filename'。(注意：kwargs 主要是用来做反向兼容的)。
      - **model_filename** (str) - 自定义 model_filename。
      - **params_filename** (str) - 自定义 params_filename。

返回：该接口返回一个包含三个元素的列表 [program，feed_target_names, fetch_targets]。它们的含义描述如下：
  - **program** （Program）– ``Program`` （详见 :ref:`api_guide_Program` ）类的实例。此处它被用于预测，因此可被称为Inference Program。
  - **feed_target_names** （list）– 字符串列表，包含着Inference Program预测时所需提供数据的所有变量名称（即所有输入变量的名称）。
  - **fetch_targets** （list）– ``Variable`` （详见 :ref:`api_guide_Program` ）类型列表，包含着模型的所有输出变量。通过这些输出变量即可得到模型的预测结果。

**返回类型：** 列表（list）

抛出异常：
  - ``ValueError`` – 如果 ``path_prefix.pdmodel`` 或 ``path_prefix.pdiparams`` 不存在，则抛出异常。

**代码示例**

.. code-block:: python

    import paddle
    import numpy as np

    paddle.enable_static()

    # 构建模型
    startup_prog = paddle.static.default_startup_program()
    main_prog = paddle.static.default_main_program()
    with paddle.static.program_guard(main_prog, startup_prog):
        image = paddle.static.data(name="img", shape=[64, 784])
        w = paddle.create_parameter(shape=[784, 200], dtype='float32')
        b = paddle.create_parameter(shape=[200], dtype='float32')
        hidden_w = paddle.matmul(x=image, y=w)
        hidden_b = paddle.add(hidden_w, b)
    exe = paddle.static.Executor(paddle.CPUPlace())
    exe.run(startup_prog)

    # 保存预测模型
    path_prefix = "./infer_model"
    paddle.static.save_inference_model(path_prefix, [image], [hidden_b], exe)

    [inference_program, feed_target_names, fetch_targets] = (
        paddle.static.load_inference_model(path_prefix, exe))
    tensor_img = np.array(np.random.random((64, 784)), dtype=np.float32)
    results = exe.run(inference_program,
                  feed={feed_target_names[0]: tensor_img},
                  fetch_list=fetch_targets)

    # 在上述示例中，inference program 被保存在 "./infer_model.pdmodel" 文件中，
    # 参数被保存在 "./infer_model.pdiparams" 文件中。
    # 加载 inference program 后， executor可使用 fetch_targets 和 feed_target_names,
    # 执行Program，并得到预测结果。

