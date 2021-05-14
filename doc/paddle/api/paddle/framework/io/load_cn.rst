.. _cn_api_paddle_framework_io_load:

load
-----

.. py:function:: paddle.load(path, **configs)

从指定路径载入可以在paddle中使用的对象实例。

.. note::
    目前支持载入：Layer 或者 Optimizer 的 ``state_dict``，Layer对象，Tensor以及包含Tensor的嵌套list、tuple、dict，Program。


如果想进一步了解这个API，请参考：

    ..  toctree::
        :maxdepth: 1
        
        ../../../../faq/save_cn.md

参数
:::::::::
    - path (str) – 载入目标对象实例的路径。通常该路径是目标文件的路径，当从用于存储预测模型API的存储结果中载入state_dict时，该路径可能是一个文件前缀或者目录。
    - **config (dict, 可选) - 其他用于兼容的载入配置选项。这些选项将来可能被移除，如果不是必须使用，不推荐使用这些配置选项。默认为 ``None``。目前支持以下配置选项：(1) model_filename (str) - paddle 1.x版本 ``save_inference_model`` 接口存储格式的预测模型文件名，原默认文件名为 ``__model__`` ； (2) params_filename (str) - paddle 1.x版本 ``save_inference_model`` 接口存储格式的参数文件名，没有默认文件名，默认将各个参数分散存储为单独的文件； (3) return_numpy(bool) - 如果被指定为 ``True`` ，``load`` 的结果中的Tensor会被转化为 ``numpy.ndarray`` ，默认为 ``False`` 。

返回
:::::::::
Object，一个可以在paddle中使用的对象实例
  
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

    # load state_dict of emb
    load_layer_state_dict = paddle.load("emb.pdparams")
    # load state_dict of optimizer
    load_opt_state_dict = paddle.load("adam.pdopt")
    # load weight of emb
    load_weight = paddle.load("emb.weight.pdtensor")

.. code-block:: python

    # example 2: Load multiple state_dict at the same time
    import paddle
    from paddle import nn
    from paddle.optimizer import Adam

    layer = paddle.nn.Linear(3, 4)
    adam = Adam(learning_rate=0.001, parameters=layer.parameters())
    obj = {'model': layer.state_dict(), 'opt': adam.state_dict(), 'epoch': 100}
    path = 'example/model.pdparams'
    paddle.save(obj, path)
    obj_load = paddle.load(path)


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
    load_tensor = paddle.load(path_tensor)

    # save/load state_dict
    path_state_dict = 'temp/model.pdparams'
    paddle.save(prog.state_dict("param"), path_tensor)
    load_state_dict = paddle.load(path_tensor)

.. code-block:: python

    # example 4: load program
    import paddle

    paddle.enable_static()

    data = paddle.static.data(
        name='x_static_save', shape=(None, 224), dtype='float32')
    y_static = z = paddle.static.nn.fc(data, 10)
    main_program = paddle.static.default_main_program()
    path = "example/main_program.pdmodel"
    paddle.save(main_program, path)
    load_main = paddle.load(path)
    print(load_main)
