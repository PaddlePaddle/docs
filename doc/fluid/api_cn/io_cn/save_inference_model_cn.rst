.. _cn_api_fluid_io_save_inference_model:

save_inference_model
-------------------------------

.. py:function:: paddle.fluid.io.save_inference_model(dirname, feeded_var_names, target_vars, executor, main_program=None, model_filename=None, params_filename=None, export_for_deployment=True,  program_only=False)

修剪指定的 ``main_program`` 以构建一个专门用于预测的 ``Inference Program`` （ ``Program`` 含义详见 :ref:`api_guide_Program` ）。 所得到的 ``Inference Program`` 及其对应的所有相关参数均被保存到 ``dirname`` 指定的目录中。若只想保存训练后的模型参数，请使用 :ref:`cn_api_fluid_io_save_params` 接口。更多细节请参考 :ref:`api_guide_model_save_reader` 。

**注意：dirname用于指定保存预测模型结构和参数的目录。若需要将模型参数保存在指定目录的若干文件中，请设置params_filename的值为None; 若需要将所有模型参数保存在一个单独的二进制文件中，请使用params_filename来指定该二进制文件的名称。**

参数:
  - **dirname** (str) – 指定保存预测模型结构和参数的文件目录。
  - **feeded_var_names** (list[str]) – 字符串列表，包含着Inference Program预测时所需提供数据的所有变量名称（即所有输入变量的名称）。
  - **target_vars** (list[Variable]) – ``Variable`` （详见 :ref:`api_guide_Program` ）类型列表，包含着模型的所有输出变量。通过这些输出变量即可得到模型的预测结果。
  - **executor** (Executor) –  用于保存预测模型的 ``executor`` ，详见 :ref:`api_guide_executor` 。
  - **main_program** (Program，可选) – 通过该参数指定的 ``main_program`` 可构建一个专门用于预测的 ``Inference Program`` 。 若为None, 则使用全局默认的  ``_main_program_`` 。默认值为None。
  - **model_filename** (str，可选) – 保存预测模型结构 ``Inference Program`` 的文件名称。若设置为None，则使用 ``__model__`` 作为默认的文件名。
  - **params_filename** (str，可选) – 保存预测模型所有相关参数的文件名称。若设置为None，则模型参数被保存在单独的文件中。
  - **export_for_deployment** (bool，可选) – 若为True，则 ``main_program`` 指定的Program将被修改为只支持直接预测部署的Program。否则，将存储更多的信息，方便优化和再训练。目前只支持设置为True，且默认值为True。
  - **program_only** (bool，可选) – 若为True，则只保存预测模型的网络结构，而不保存预测模型的网络参数。默认值为False。


**返回：** 用于获取模型预测结果的所有输出变量的名称列表。

**返回类型：** 列表（list）

抛出异常：
 - ``ValueError`` – 若 ``feed_var_names`` 不是字符串列表，则抛出异常。
 - ``ValueError`` – 若 ``target_vars`` 不是 ``Variable`` 类型列表，则抛出异常。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid

    path = "./infer_model"

    # 用户定义网络，此处以softmax回归为例
    image = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    feeder = fluid.DataFeeder(feed_list=[image, label], place=fluid.CPUPlace())
    predict = fluid.layers.fc(input=image, size=10, act='softmax')

    loss = fluid.layers.cross_entropy(input=predict, label=label)
    avg_loss = fluid.layers.mean(loss)

    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())

    # 数据输入及训练过程

    # 保存预测模型。注意，用于预测的模型网络结构不需要保存标签和损失。
    fluid.io.save_inference_model(dirname=path, feeded_var_names=['img'], target_vars=[predict], executor=exe)

    # 在这个示例中，save_inference_mode接口将根据网络的输入结点（img）和输出结点（predict）修剪默认的主程序（_main_program_）。
    # 修剪得到的Inference Program将被保存在 “./infer_model/__model__”文件中，
    # 模型参数将被保存在“./infer_model/”文件夹下以各自名称命名的单独文件中。








