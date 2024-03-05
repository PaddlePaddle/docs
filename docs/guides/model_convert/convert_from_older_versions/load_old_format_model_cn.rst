.. _cn_guides_load_old_format_model:


兼容载入旧格式模型
====================

如果你是从飞桨框架 1.x 切换到 2.1，曾经使用飞桨框架 1.x 的 fluid 相关接口保存模型或者参数，飞桨框架 2.1 也对这种情况进行了兼容性支持，包括以下几种情况。

飞桨 1.x 模型准备及训练示例，该示例为后续所有示例的前序逻辑：

.. code-block:: python

    import numpy as np
    import paddle
    import paddle.fluid as fluid
    import paddle.nn as nn
    import paddle.optimizer as opt

    BATCH_SIZE = 16
    BATCH_NUM = 4
    EPOCH_NUM = 4

    IMAGE_SIZE = 784
    CLASS_NUM = 10

    # enable static graph mode
    paddle.enable_static()

    # define a random dataset
    class RandomDataset(paddle.io.Dataset):
        def __init__(self, num_samples):
            self.num_samples = num_samples

        def __getitem__(self, idx):
            image = np.random.random([IMAGE_SIZE]).astype('float32')
            label = np.random.randint(0, CLASS_NUM - 1, (1, )).astype('int64')
            return image, label

        def __len__(self):
            return self.num_samples

    image = fluid.data(name='image', shape=[None, 784], dtype='float32')
    label = fluid.data(name='label', shape=[None, 1], dtype='int64')
    pred = fluid.layers.fc(input=image, size=10, act='softmax')
    loss = fluid.layers.cross_entropy(input=pred, label=label)
    avg_loss = fluid.layers.mean(loss)

    optimizer = fluid.optimizer.SGD(learning_rate=0.001)
    optimizer.minimize(avg_loss)

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    # create data loader
    dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
    loader = paddle.io.DataLoader(dataset,
        feed_list=[image, label],
        places=place,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=2)

    # train model
    for data in loader():
        exe.run(
            fluid.default_main_program(),
            feed=data,
            fetch_list=[avg_loss])


1 从 ``paddle.fluid.io.save_inference_model`` 保存结果中载入模型&参数
----------------------------------------------------------------------------

1.1 同时载入模型和参数
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

使用 ``paddle.jit.load`` 配合 ``**configs`` 载入模型和参数。

如果你是按照 ``paddle.fluid.io.save_inference_model`` 的默认格式存储的，可以按照如下方式载入（接前述示例）：

.. code-block:: python

    # save default
    model_path = "fc.example.model"
    fluid.io.save_inference_model(
        model_path, ["image"], [pred], exe)

    # enable dynamic mode
    paddle.disable_static(place)

    # load
    fc = paddle.jit.load(model_path)

    # inference
    fc.eval()
    x = paddle.randn([1, IMAGE_SIZE], 'float32')
    pred = fc(x)

如果你指定了存储的模型文件名，可以按照以下方式载入（接前述示例）：

.. code-block:: python

    # save with model_filename
    model_path = "fc.example.model.with_model_filename"
    fluid.io.save_inference_model(
        model_path, ["image"], [pred], exe, model_filename="__simplenet__")

    # enable dynamic mode
    paddle.disable_static(place)

    # load
    fc = paddle.jit.load(model_path, model_filename="__simplenet__")

    # inference
    fc.eval()
    x = paddle.randn([1, IMAGE_SIZE], 'float32')
    pred = fc(x)

如果你指定了存储的参数文件名，可以按照以下方式载入（接前述示例）：

.. code-block:: python

    # save with params_filename
    model_path = "fc.example.model.with_params_filename"
    fluid.io.save_inference_model(
        model_path, ["image"], [pred], exe, params_filename="__params__")

    # enable dynamic mode
    paddle.disable_static(place)

    # load
    fc = paddle.jit.load(model_path, params_filename="__params__")

    # inference
    fc.eval()
    x = paddle.randn([1, IMAGE_SIZE], 'float32')
    pred = fc(x)

1.2 仅载入参数
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
如果你仅需要从 ``paddle.fluid.io.save_inference_model`` 的存储结果中载入参数，以 state_dict 的形式配置到已有代码的模型中，可以使用 ``paddle.load`` 配合 ``**configs`` 载入。

如果你是按照 ``paddle.fluid.io.save_inference_model`` 的默认格式存储的，可以按照如下方式载入（接前述示例）：

.. code-block:: python

    model_path = "fc.example.model"

    load_param_dict = paddle.load(model_path)

如果你指定了存储的模型文件名，可以按照以下方式载入（接前述示例）：

.. code-block:: python

    model_path = "fc.example.model.with_model_filename"

    load_param_dict = paddle.load(model_path, model_filename="__simplenet__")

如果你指定了存储的参数文件名，可以按照以下方式载入（接前述示例）：

.. code-block:: python

    model_path = "fc.example.model.with_params_filename"

    load_param_dict = paddle.load(model_path, params_filename="__params__")

.. note::
    一般预测模型不会存储优化器 Optimizer 的参数，因此此处载入的仅包括模型本身的参数。

.. note::
    由于 ``structured_name`` 是动态图下独有的变量命名方式，因此从静态图存储结果载入的 state_dict 在配置到动态图的 Layer 中时，需要配置 ``Layer.set_state_dict(use_structured_name=False)`` 。


2 从 ``paddle.fluid.save`` 存储结果中载入参数
----------------------------------------------------------------------------

 ``paddle.fluid.save`` 的存储格式与 2.x 动态图接口 ``paddle.save`` 存储格式是类似的，同样存储了 dict 格式的参数，因此可以直接使用 ``paddle.load`` 载入 state_dict，但需要注意不能仅传入保存的路径，而要传入保存参数的文件名，示例如下（接前述示例）：

.. code-block:: python

    # save by fluid.save
    model_path = "fc.example.model.save"
    program = fluid.default_main_program()
    fluid.save(program, model_path)

    # enable dynamic mode
    paddle.disable_static(place)

    load_param_dict = paddle.load("fc.example.model.save.pdparams")


.. note::
    由于 ``paddle.fluid.save`` 接口原先在静态图模式下的定位是存储训练时参数，或者说存储 Checkpoint，故尽管其同时存储了模型结构，目前也暂不支持从 ``paddle.fluid.save`` 的存储结果中同时载入模型和参数，后续如有需求再考虑支持。


3 从 ``paddle.fluid.io.save_params/save_persistables`` 保存结果中载入参数
----------------------------------------------------------------------------

这两个接口在飞桨 1.x 版本时，已经不再推荐作为存储模型参数的接口使用，故并未继承至飞桨 2.x，之后也不会再推荐使用这两个接口存储参数。

对于使用这两个接口存储参数兼容载入的支持，分为两种情况，下面以 ``paddle.fluid.io.save_params`` 接口为例介绍相关使用方法：


3.1 使用默认方式存储，各参数分散存储为单独的文件，文件名为参数名
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

这种存储方式仍然可以使用 ``paddle.load`` 接口兼容载入，使用示例如下（接前述示例）：

.. code-block:: python

    # save by fluid.io.save_params
    model_path = "fc.example.model.save_params"
    fluid.io.save_params(exe, model_path)

    # load
    state_dict = paddle.load(model_path)
    print(state_dict)

3.2 指定了参数存储的文件，将所有参数存储至单个文件中
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
将所有参数存储至单个文件中会导致存储结果中丢失 Tensor 名和 Tensor 数据之间的映射关系，因此这部分丢失的信息需要用户传入进行补足。为了确保正确性，这里不仅要传入 Tensor 的 name 列表，同时要传入 Tensor 的 shape 和 dtype 等描述信息，通过检查和存储数据的匹配性确保严格的正确性，这导致载入数据的恢复过程变得比较复杂，仍然需要一些飞桨 1.x 的概念支持。后续如果此项需求较为普遍，飞桨将会考虑将该项功能兼容支持到 ``paddle.load`` 中，但由于信息丢失而导致的使用复杂性仍然是存在的，因此建议你避免仅使用这两个接口存储参数。

目前暂时推荐你使用 ``paddle.static.load_program_state`` 接口解决此处的载入问题，需要获取原 Program 中的参数列表传入该方法，使用示例如下（接前述示例）：

.. code-block:: python

    # save by fluid.io.save_params
    model_path = "fc.example.model.save_params_with_filename"
    fluid.io.save_params(exe, model_path, filename="__params__")

    # load
    import os
    params_file_path = os.path.join(model_path, "__params__")
    var_list = fluid.default_main_program().all_parameters()
    state_dict = paddle.io.load_program_state(params_file_path, var_list)


4 从 ``paddle.static.save`` 保存结果中载入参数
----------------------------------------------------------------------------
``paddle.static.save`` 接口生成三个文件： ``*.pdparams`` 、 ``*.pdopt`` 、 ``*.pdmodel`` ，分别保存了组网的参数、优化器的参数、静态图的 Program。推荐您使用 ``paddle.load`` 分别加载这三个文件，然后使用 ``set_state_dict`` 接口将参数设置到 ``Program`` 中 。如果您已经在代码中定义了 ``Program`` ，您可以不加载 ``*.pdmodel`` 文件；如果您不需要恢复优化器中的参数，您可以不加载 ``*.pdopt`` 文件。使用示例如下：


.. code-block:: python

    import os
    import paddle

    paddle.enable_static()
    x = paddle.static.data(
                    name="static_x", shape=[None, 224], dtype='float32')
    z = paddle.static.nn.fc(x, 10)
    z = paddle.static.nn.fc(z, 10, bias_attr=False)

    place = paddle.CPUPlace()
    exe = paddle.static.Executor(place)
    exe.run(paddle.static.default_startup_program())
    prog = paddle.static.default_main_program()

    path = os.path.join("test_static_save_load", "model")
    paddle.static.save(prog, path)

    # load program
    program=paddle.load(path + '.pdmodel')

    state_dict_param = paddle.load(path + '.pdparams')
    program.set_state_dict(state_dict_param)

    state_dict_opt = paddle.load(path + '.pdopt')
    program.set_state_dict(state_dict_opt)
