.. _cn_api_fluid_io_save_inference_model:

save_inference_model
-------------------------------

.. py:function:: paddle.fluid.io.save_inference_model(dirname, feeded_var_names, target_vars, executor, main_program=None, model_filename=None, params_filename=None, export_for_deployment=True,  program_only=False)

修改指定的 ``main_program`` ，构建一个专门用于预测的 ``Program``，然后  ``executor`` 把它和所有相关参数保存到 ``dirname`` 中。


``dirname`` 用于指定保存变量的目录。如果变量保存在指定目录的若干文件中，设置文件名 None; 如果所有变量保存在一个文件中，请使用filename来指定它。

如果您仅想保存您训练好的模型的参数，请使用save_params API。更多细节请参考 :ref:`api_guide_model_save_reader` 。


参数:
  - **dirname** (str) – 保存预测model的路径
  - **feeded_var_names** (list[str]) – 预测（inference）需要 feed 的数据
  - **target_vars** (list[Variable]) – 保存预测（inference）结果的 Variables
  - **executor** (Executor) –  executor 保存  inference model
  - **main_program** (Program|None) – 使用 ``main_program`` ，构建一个专门用于预测的 ``Program`` （inference model）. 如果为None, 使用   ``default main program``   默认: None.
  - **model_filename** (str|None) – 保存预测Program 的文件名称。如果设置为None，将使用默认的文件名为： ``__model__``
  - **params_filename** (str|None) – 保存所有相关参数的文件名称。如果设置为None，则参数将保存在单独的文件中。
  - **export_for_deployment** (bool) – 如果为真，Program将被修改为只支持直接预测部署的Program。否则，将存储更多的信息，方便优化和再训练。目前只支持True。
  - **program_only** (bool) – 如果为真，将只保存预测程序，而不保存程序的参数。

返回: 获取的变量名列表

返回类型：target_var_name_list(list)

抛出异常：
 - ``ValueError`` – 如果 ``feed_var_names`` 不是字符串列表
 - ``ValueError`` – 如果 ``target_vars`` 不是 ``Variable`` 列表

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

    # 保存预测模型。注意我们不在这个示例中保存标签和损失。
    fluid.io.save_inference_model(dirname=path, feeded_var_names=['img'], target_vars=[predict], executor=exe)

    # 在这个示例中，函数将修改默认的主程序让它适合于预测‘predict_var’
    # 修改的预测Program 将被保存在 ./infer_model/__model__”中。
    # 参数将保存在文件夹下的单独文件中 ./infer_mode








