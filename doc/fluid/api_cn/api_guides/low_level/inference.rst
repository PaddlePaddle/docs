..  _api_guide_inference:

#########
预测引擎
#########

预测引擎提供了存储预测模型 :ref:`api_fluid_io_save_inference_model` 和加载预测模型 :ref:`api_fluid_io_load_inference_model` 两个接口。

预测模型的存储格式
=================

预测模型的存储格式有两种，由上述两个接口中的 :code:`model_filename` 和 :code:`params_filename` 变量控制：

- 参数保存到各个独立的文件，如设置 :code:`model_filename` 为 :code:`None` 、:code:`params_filename` 为 :code:`None`

  .. code-block:: bash

      ls recognize_digits_conv.inference.model/*
      __model__ conv2d_1.w_0 conv2d_2.w_0 fc_1.w_0 conv2d_1.b_0 conv2d_2.b_0 fc_1.b_0

- 参数保存到同一个文件，如设置 :code:`model_filename` 为 :code:`None` 、:code:`params_filename` 为 :code:`__params__`

  .. code-block:: bash

      ls recognize_digits_conv.inference.model/*
      __model__ __params__

存储预测模型
===========

.. code-block:: python

    exe = fluid.Executor(fluid.CPUPlace())
    path = "./infer_model"
    fluid.io.save_inference_model(dirname=path, feeded_var_names=['img'], 
        target_vars=[predict_var], executor=exe)

在这个示例中，:code:`fluid.io.save_inference_model` 接口对默认的 :code:`fluid.Program` 进行裁剪，只保留预测 :code:`predict_var` 所需部分。
裁剪后的 :code:`program` 会保存在 :code:`./infer_model/__model__` 下，参数会保存到 :code:`./infer_model` 下的各个独立文件。

加载预测模型
===========

.. code-block:: python

    exe = fluid.Executor(fluid.CPUPlace())
    path = "./infer_model"
    [inference_program, feed_target_names, fetch_targets] = 
        fluid.io.load_inference_model(dirname=path, executor=exe)
    results = exe.run(inference_program,
                  feed={feed_target_names[0]: tensor_img},
                  fetch_list=fetch_targets)

在这个示例中，首先调用 :code:`fluid.io.load_inference_model` 接口，获得预测的 :code:`program` 、输入数据的 :code:`variable` 名称和输出结果的 :code:`variable` ;
然后调用 :code:`executor` 执行预测的 :code:`program` 获得预测结果。
