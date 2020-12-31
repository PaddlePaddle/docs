.. _cn_api_static_save_inference_model:

save_inference_model
-------------------------------


.. py:function:: paddle.static.save_inference_model(path_prefix, feed_vars, fetch_vars, executor, **kwargs)




将模型及其参数保存到指定的路径。例如，``path_prefix="/path/to/modelname"`` ，在调用 ``save_inference_model(path_prefix, feed_vars, fetch_vars, executor)`` 之后，你可以在 "/path/to" 目录下找到两个文件，分别是 "modelname.pdmodel" 和 "modelname.pdiparams"，前者表示序列化之后的模型文件，后者表示序列化之后的参数文件。


参数:
  - **path_prefix** (str) – 要保存到的目录 + 模型名称（不包含后缀）。
  - **feed_vars** (Variable | list[Variable]) – 模型的所有输入变量。
  - **fetch_vars** (Variable | list[Variable]) – 模型的所有输出变量。
  - **executor** (Executor) –  用于保存预测模型的 ``executor`` ，详见 :ref:`api_guide_executor` 。
  - **kwargs** - 支持的 key 包括 'program'。(注意：kwargs 主要是用来做反向兼容的)。
      - **program** - 自定义一个 program，默认使用 default_main_program。


**返回：** None

抛出异常：
 - ``ValueError`` – 若 ``feed_vars`` 不是 ``Variable`` 或 ``Variable`` 类型列表，则抛出异常。
 - ``ValueError`` – 若 ``fetch_vars`` 不是 ``Variable`` 或 ``Variable`` 类型列表，则抛出异常。

**代码示例**

.. code-block:: python

    import paddle

    paddle.enable_static()

    path_prefix = "./infer_model"

    # 用户定义网络，此处以softmax回归为例
    image = paddle.static.data(name='img', shape=[None, 28, 28], dtype='float32')
    label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
    predict = paddle.static.nn.fc(image, 10, activation='softmax')

    loss = paddle.nn.functional.cross_entropy(predict, label)
    avg_loss = paddle.tensor.stat.mean(loss)

    exe = paddle.static.Executor(paddle.CPUPlace())
    exe.run(paddle.static.default_startup_program())

    # 数据输入及训练过程

    # 保存预测模型。注意，用于预测的模型网络结构不需要保存标签和损失。
    paddle.static.save_inference_model(path_prefix, [image], [predict], exe)

    # 在本示例中，save_inference_mode 将根据网络的输入（image）和输出（predict）修剪模型。
    # 修剪得到的模型将被保存在 "./infer_model.pdmodel" 文件中，
    # 模型参数将被保存在 "./infer_model.pdiparams" 文件中。

