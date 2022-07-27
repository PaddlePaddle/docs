.. _cn_api_paddle_framework_io_save:

save
-----

.. py:function:: paddle.save(obj, path, protocol=4)

将对象实例 obj 保存到指定的路径中。

.. note::
    目前支持保存：Layer 或者 Optimizer 的 ``state_dict``，Tensor 以及包含 Tensor 的嵌套 list、tuple、dict，Program。对于 Tensor 对象，只保存了它的名字和数值，没有保存 stop_gradient 等属性，如果您需要这些没有保存的属性，请调用 set_value 接口将数值设置到带有这些属性的 Tensor 中。

.. note::
    不同于 ``paddle.jit.save``，由于 ``paddle.save`` 的存储结果是单个文件，所以不需要通过添加后缀的方式区分多个存储文件，``paddle.save`` 的输入参数 ``path`` 将直接作为存储结果的文件名而非前缀。为了统一存储文件名的格式，我们推荐使用 paddle 标椎文件后缀：
    1. 对于 ``Layer.state_dict``，推荐使用后缀 ``.pdparams`` ；
    2. 对于 ``Optimizer.state_dict``，推荐使用后缀 ``.pdopt`` 。
    具体示例请参考 API 的代码示例。


遇到使用问题，请参考：

    ..  toctree::
        :maxdepth: 1

        ../../../../faq/save_cn.md

参数
:::::::::
 - **obj**  (Object) – 要保存的对象实例。
 - **path**  (str|BytesIO) – 保存对象实例的路径/内存对象。如果存储到当前路径，输入的 path 字符串将会作为保存的文件名。
 - **protocol**  (int，可选) – pickle 模块的协议版本，默认值为 4，取值范围是[2,4]。
 - **configs**  (dict，可选) – 其他配置选项，目前支持以下选项：（1）use_binary_format（bool）- 如果被保存的对象是静态图的 Tensor，你可以指定这个参数。如果被指定为 ``True``，这个 Tensor 会被保存为由 paddle 定义的二进制格式的文件；否则这个 Tensor 被保存为 pickle 格式。默认为 ``False`` 。

返回
:::::::::
无

代码示例
:::::::::

.. code-block:: python

    # example 1: dynamic graph
    import paddle
    emb = paddle.nn.Embedding(10, 10)
    layer_state_dict = emb.state_dict()

    # save state_dict of emb
    paddle.save(layer_state_dict, "emb.pdparams")

    scheduler = paddle.optimizer.lr.NoamDecay(
        d_model=0.01, warmup_steps=100, verbose=True)
    adam = paddle.optimizer.Adam(
        learning_rate=scheduler,
        parameters=emb.parameters())
    opt_state_dict = adam.state_dict()

    # save state_dict of optimizer
    paddle.save(opt_state_dict, "adam.pdopt")
    # save weight of emb
    paddle.save(emb.weight, "emb.weight.pdtensor")


.. code-block:: python

    # example 2: Save multiple state_dict at the same time
    import paddle
    from paddle import nn
    from paddle.optimizer import Adam

    layer = paddle.nn.Linear(3, 4)
    adam = Adam(learning_rate=0.001, parameters=layer.parameters())
    obj = {'model': layer.state_dict(), 'opt': adam.state_dict(), 'epoch': 100}
    path = 'example/model.pdparams'
    paddle.save(obj, path)


.. code-block:: python

    # example 3: static graph
    import paddle
    import paddle.static as static

    paddle.enable_static()

    # create network
    x = paddle.static.data(name="x", shape=[None, 224], dtype='float32')
    z = paddle.static.nn.fc(x, 10)

    place = paddle.CPUPlace()
    exe = paddle.static.Executor(place)
    exe.run(paddle.static.default_startup_program())
    prog = paddle.static.default_main_program()
    for var in prog.list_vars():
        if list(var.shape) == [224, 10]:
            tensor = var.get_value()
            break

    # save/load tensor
    path_tensor = 'temp/tensor.pdtensor'
    paddle.save(tensor, path_tensor)

    # save/load state_dict
    path_state_dict = 'temp/model.pdparams'
    paddle.save(prog.state_dict("param"), path_tensor)


.. code-block:: python

    # example 4: save program
    import paddle

    paddle.enable_static()

    data = paddle.static.data(
        name='x_static_save', shape=(None, 224), dtype='float32')
    y_static = z = paddle.static.nn.fc(data, 10)
    main_program = paddle.static.default_main_program()
    path = "example/main_program.pdmodel"
    paddle.save(main_program, path)
