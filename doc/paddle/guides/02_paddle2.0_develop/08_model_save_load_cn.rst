.. _cn_doc_model_save_load:

#############
模型存储与载入
#############

一、存储载入体系简介
##################

1.1 接口体系
------------

飞桨框架2.x对模型与参数的存储与载入相关接口进行了梳理，根据接口使用的场景与模式，分为三套体系，分别是：

1.1.1 动态图存储载入体系
```````````````````````

为提升框架使用体验，飞桨框架2.0将主推动态图模式，动态图模式下的存储载入接口包括：

- paddle.save
- paddle.load
- paddle.jit.save
- paddle.jit.load

本文主要介绍飞桨框架2.0动态图存储载入体系，各接口关系如下图所示：

.. image:: https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/paddle/guides/images/main_save_load_2.0.png?raw=true

1.1.2 静态图存储载入体系
```````````````````````

静态图存储载入相关接口为飞桨框架1.x版本的主要使用接口，出于兼容性的目的，这些接口仍然可以在飞桨框架2.x使用，但不再推荐。相关接口包括：

- paddle.static.save
- paddle.static.load
- paddle.static.save_inference_model
- paddle.static.load_inference_model
- paddle.static.load_program_state
- paddle.static.set_program_state

由于飞桨框架2.0不再主推静态图模式，故本文不对以上主要用于飞桨框架1.x的相关接口展开介绍，如有需要，可以阅读对应API文档。

1.1.3 高阶API存储载入体系
````````````````````````

- paddle.Model.fit (训练接口，同时带有参数存储的功能)
- paddle.Model.save
- paddle.Model.load

飞桨框架2.0高阶API仅有一套Save/Load接口，表意直观，体系清晰，若有需要，建议直接阅读相关API文档，此处不再赘述。

.. note::
    本教程着重介绍飞桨框架2.x的各个存储载入接口的关系及各种使用场景，不对接口参数进行详细介绍，如果需要了解具体接口参数的含义，请直接阅读对应API文档。

1.2 接口存储结果组织形式
----------------------

飞桨2.0统一了各存储接口对于同一种存储行为的处理方式，并且统一推荐或自动为存储的文件添加飞桨标准的文件后缀，详见下图：

.. image:: https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/paddle/guides/images/save_result_format_2.0.png?raw=true


二、参数存储载入（训练调优）
#######################

若仅需要存储/载入模型的参数，可以使用 ``paddle.save/load`` 结合Layer和Optimizer的state_dict达成目的，此处state_dict是对象的持久参数的载体，dict的key为参数名，value为参数真实的numpy array值。

结合以下简单示例，介绍参数存储和载入的方法，以下示例完成了一个简单网络的训练过程：

.. code-block:: python

    import numpy as np
    import paddle
    import paddle.nn as nn
    import paddle.optimizer as opt

    BATCH_SIZE = 16
    BATCH_NUM = 4
    EPOCH_NUM = 4

    IMAGE_SIZE = 784
    CLASS_NUM = 10

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

    class LinearNet(nn.Layer):
        def __init__(self):
            super(LinearNet, self).__init__()
            self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)

        def forward(self, x):
            return self._linear(x)

    def train(layer, loader, loss_fn, opt):
        for epoch_id in range(EPOCH_NUM):
            for batch_id, (image, label) in enumerate(loader()):
                out = layer(image)
                loss = loss_fn(out, label)
                loss.backward()
                opt.step()
                opt.clear_grad()
                print("Epoch {} batch {}: loss = {}".format(
                    epoch_id, batch_id, np.mean(loss.numpy())))

    # create network
    layer = LinearNet()
    loss_fn = nn.CrossEntropyLoss()
    adam = opt.Adam(learning_rate=0.001, parameters=layer.parameters())

    # create data loader
    dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
    loader = paddle.io.DataLoader(dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=2)

    # train
    train(layer, loader, loss_fn, adam)


2.1 参数存储
------------

参数存储时，先获取目标对象（Layer或者Optimzier）的state_dict，然后将state_dict存储至磁盘，示例如下（接前述示例）:

.. code-block:: python

    # save
    paddle.save(layer.state_dict(), "linear_net.pdparams")
    paddle.save(adam.state_dict(), "adam.pdopt")


2.2 参数载入
------------

参数载入时，先从磁盘载入保存的state_dict，然后通过set_state_dict方法配置到目标对象中，示例如下（接前述示例）：

.. code-block:: python

    # load
    layer_state_dict = paddle.load("linear_net.pdparams")
    opt_state_dict = paddle.load("adam.pdopt")

    layer.set_state_dict(layer_state_dict)
    adam.set_state_dict(opt_state_dict)


三、模型&参数存储载入（训练部署）
############################

若要同时存储/载入模型结构和参数，可以使用 ``paddle.jit.save/load`` 实现。

3.1 模型&参数存储
----------------

模型&参数存储根据训练模式不同，有两种使用情况：

(1) 动转静训练 + 模型&参数存储
(2) 动态图训练 + 模型&参数存储

3.1.1 动转静训练 + 模型&参数存储
``````````````````````````````

动转静训练相比直接使用动态图训练具有更好的执行性能，训练完成后，直接将目标Layer传入 ``paddle.jit.save`` 存储即可。：

一个简单的网络训练示例如下：

.. code-block:: python

    import numpy as np
    import paddle
    import paddle.nn as nn
    import paddle.optimizer as opt

    BATCH_SIZE = 16
    BATCH_NUM = 4
    EPOCH_NUM = 4

    IMAGE_SIZE = 784
    CLASS_NUM = 10

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

    class LinearNet(nn.Layer):
        def __init__(self):
            super(LinearNet, self).__init__()
            self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)

        @paddle.jit.to_static
        def forward(self, x):
            return self._linear(x)

    def train(layer, loader, loss_fn, opt):
        for epoch_id in range(EPOCH_NUM):
            for batch_id, (image, label) in enumerate(loader()):
                out = layer(image)
                loss = loss_fn(out, label)
                loss.backward()
                opt.step()
                opt.clear_grad()
                print("Epoch {} batch {}: loss = {}".format(
                    epoch_id, batch_id, np.mean(loss.numpy())))

    # create network
    layer = LinearNet()
    loss_fn = nn.CrossEntropyLoss()
    adam = opt.Adam(learning_rate=0.001, parameters=layer.parameters())

    # create data loader
    dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
    loader = paddle.io.DataLoader(dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=2)

    # train
    train(layer, loader, loss_fn, adam)


随后使用 ``paddle.jit.save`` 对模型和参数进行存储（接前述示例）：

.. code-block:: python

    # save
    path = "example.model/linear"
    paddle.jit.save(layer, path)


通过动转静训练后保存模型&参数，有以下三项注意点：

(1) Layer对象的forward方法需要经由 ``paddle.jit.to_static`` 装饰

经过 ``paddle.jit.to_static`` 装饰forward方法后，相应Layer在执行时，会先生成描述模型的Program，然后通过执行Program获取计算结果，示例如下：

.. code-block:: python

    import paddle
    import paddle.nn as nn

    IMAGE_SIZE = 784
    CLASS_NUM = 10

    class LinearNet(nn.Layer):
        def __init__(self):
            super(LinearNet, self).__init__()
            self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)

        @paddle.jit.to_static
        def forward(self, x):
            return self._linear(x)

若最终需要生成的描述模型的Program支持动态输入，可以同时指明模型的 ``InputSepc`` ，示例如下：

.. code-block:: python

    import paddle
    import paddle.nn as nn
    from paddle.static import InputSpec

    IMAGE_SIZE = 784
    CLASS_NUM = 10

    class LinearNet(nn.Layer):
        def __init__(self):
            super(LinearNet, self).__init__()
            self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)

        @paddle.jit.to_static(input_spec=[InputSpec(shape=[None, 784], dtype='float32')])
        def forward(self, x):
            return self._linear(x)


(2) 请确保Layer.forward方法中仅实现预测功能，避免将训练所需的loss计算逻辑写入forward方法

Layer更准确的语义是描述一个具有预测功能的模型对象，接收输入的样本数据，输出预测的结果，而loss计算是仅属于模型训练中的概念。将loss计算的实现放到Layer.forward方法中，会使Layer在不同场景下概念有所差别，并且增大Layer使用的复杂性，这不是良好的编码行为，同时也会在最终保存预测模型时引入剪枝的复杂性，因此建议保持Layer实现的简洁性，下面通过两个示例对比说明：

错误示例如下：

.. code-block:: python

    import paddle
    import paddle.nn as nn

    IMAGE_SIZE = 784
    CLASS_NUM = 10

    class LinearNet(nn.Layer):
        def __init__(self):
            super(LinearNet, self).__init__()
            self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)

        @paddle.jit.to_static
        def forward(self, x, label=None):
            out = self._linear(x)
            if label:
                loss = nn.functional.cross_entropy(out, label)
                avg_loss = nn.functional.mean(loss)
                return out, avg_loss
            else:
                return out
            

正确示例如下：

.. code-block:: python

    import paddle
    import paddle.nn as nn

    IMAGE_SIZE = 784
    CLASS_NUM = 10

    class LinearNet(nn.Layer):
        def __init__(self):
            super(LinearNet, self).__init__()
            self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)

        @paddle.jit.to_static
        def forward(self, x):
            return self._linear(x)


(3) 如果您需要存储多个方法，需要用 ``paddle.jit.to_static`` 装饰每一个需要被存储的方法。
只有在forward之外还需要存储其他方法时才用这个特性，如果仅装饰非forward的方法，而forward没有被装饰，是不符合规范的。此时 ``paddle.jit.save`` 的 ``input_spec`` 参数必须为None。示例如下：

.. code-block:: python

    import paddle
    import paddle.nn as nn
    from paddle.static import InputSpec

    IMAGE_SIZE = 784
    CLASS_NUM = 10

    class LinearNet(nn.Layer):
        def __init__(self):
            super(LinearNet, self).__init__()
            self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)
            self._linear_2 = nn.Linear(IMAGE_SIZE, CLASS_NUM)

        @paddle.jit.to_static(input_spec=[InputSpec(shape=[None, IMAGE_SIZE], dtype='float32')])
        def forward(self, x):
            return self._linear(x)

        @paddle.jit.to_static(input_spec=[InputSpec(shape=[None, IMAGE_SIZE], dtype='float32')])
        def another_forward(self, x):
            return self._linear_2(x)

    inps = paddle.randn([1, IMAGE_SIZE])
    layer = LinearNet()
    before_0 = layer.another_forward(inps)
    before_1 = layer(inps)
    # save and load
    path = "example.model/linear"
    paddle.jit.save(layer, path)

存储的模型命名规则：forward的模型名字为：模型名+后缀，其他函数的模型名字为：模型名+函数名+后缀。每个函数有各自的pdmodel和pdiparams的文件，所有函数共用pdiparams.info。上述代码将在 ``example.model`` 文件夹下产生5个文件：
``linear.another_forward.pdiparams    linear.pdiparams    linear.pdmodel    linear.another_forward.pdmodel    linear.pdiparams.info``

3.1.2 动态图训练 + 模型&参数存储
``````````````````````````````

动态图模式相比动转静模式更加便于调试，如果您仍需要使用动态图直接训练，也可以在动态图训练完成后调用 ``paddle.jit.save`` 直接存储模型和参数。

同样是一个简单的网络训练示例：

.. code-block:: python

    import numpy as np
    import paddle
    import paddle.nn as nn
    import paddle.optimizer as opt
    from paddle.static import InputSpec

    BATCH_SIZE = 16
    BATCH_NUM = 4
    EPOCH_NUM = 4

    IMAGE_SIZE = 784
    CLASS_NUM = 10

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

    class LinearNet(nn.Layer):
        def __init__(self):
            super(LinearNet, self).__init__()
            self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)

        def forward(self, x):
            return self._linear(x)

    def train(layer, loader, loss_fn, opt):
        for epoch_id in range(EPOCH_NUM):
            for batch_id, (image, label) in enumerate(loader()):
                out = layer(image)
                loss = loss_fn(out, label)
                loss.backward()
                opt.step()
                opt.clear_grad()
                print("Epoch {} batch {}: loss = {}".format(
                    epoch_id, batch_id, np.mean(loss.numpy())))

    # create network
    layer = LinearNet()
    loss_fn = nn.CrossEntropyLoss()
    adam = opt.Adam(learning_rate=0.001, parameters=layer.parameters())

    # create data loader
    dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
    loader = paddle.io.DataLoader(dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=2)

    # train
    train(layer, loader, loss_fn, adam)


训练完成后使用 ``paddle.jit.save`` 对模型和参数进行存储：

.. code-block:: python

    # save
    path = "example.dy_model/linear"
    paddle.jit.save(
        layer=layer, 
        path=path,
        input_spec=[InputSpec(shape=[None, 784], dtype='float32')])

动态图训练后使用 ``paddle.jit.save`` 存储模型和参数注意点如下：

(1) 相比动转静训练，Layer对象的forward方法不需要额外装饰，保持原实现即可

(2) 与动转静训练相同，请确保Layer.forward方法中仅实现预测功能，避免将训练所需的loss计算逻辑写入forward方法

(3) 在最后使用 ``paddle.jit.save`` 时，需要指定Layer的 ``InputSpec`` ，Layer对象forward方法的每一个参数均需要对应的 ``InputSpec`` 进行描述，不能省略。这里的 ``input_spec`` 参数支持两种类型的输入：

- ``InputSpec`` 列表

使用InputSpec描述forward输入参数的shape，dtype和name，如前述示例（此处示例中name省略，name省略的情况下会使用forward的对应参数名作为name，所以这里的name为 ``x`` ）：

.. code-block:: python

    paddle.jit.save(
        layer=layer, 
        path=path,
        input_spec=[InputSpec(shape=[None, 784], dtype='float32')])

- Example Tensor 列表

除使用InputSpec之外，也可以直接使用forward训练时的示例输入，此处可以使用前述示例中迭代DataLoader得到的 ``image`` ，示例如下：

.. code-block:: python

    paddle.jit.save(
        layer=layer, 
        path=path,
        input_spec=[image])

3.2 模型&参数载入
----------------

载入模型参数，使用 ``paddle.jit.load`` 载入即可，载入后得到的是一个Layer的派生类对象 ``TranslatedLayer`` ， ``TranslatedLayer`` 具有Layer具有的通用特征，支持切换 ``train`` 或者 ``eval`` 模式，可以进行模型调优或者预测。为了规避变量名字冲突，载入之后参数的名字可能发生变化。

载入模型及参数，示例如下：

.. code-block:: python

    import numpy as np
    import paddle
    import paddle.nn as nn
    import paddle.optimizer as opt

    BATCH_SIZE = 16
    BATCH_NUM = 4
    EPOCH_NUM = 4

    IMAGE_SIZE = 784
    CLASS_NUM = 10

    # load
    path = "example.model/linear"
    loaded_layer = paddle.jit.load(path)

载入模型及参数后进行预测，示例如下（接前述示例）：

.. code-block:: python

    # inference
    loaded_layer.eval()
    x = paddle.randn([1, IMAGE_SIZE], 'float32')
    pred = loaded_layer(x)

载入模型及参数后进行调优，示例如下（接前述示例）：

.. code-block:: python

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

    def train(layer, loader, loss_fn, opt):
        for epoch_id in range(EPOCH_NUM):
            for batch_id, (image, label) in enumerate(loader()):
                out = layer(image)
                loss = loss_fn(out, label)
                loss.backward()
                opt.step()
                opt.clear_grad()
                print("Epoch {} batch {}: loss = {}".format(
                    epoch_id, batch_id, np.mean(loss.numpy())))

    # fine-tune
    loaded_layer.train()
    dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
    loader = paddle.io.DataLoader(dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=2)
    loss_fn = nn.CrossEntropyLoss()
    adam = opt.Adam(learning_rate=0.001, parameters=loaded_layer.parameters())
    train(loaded_layer, loader, loss_fn, adam)
    # save after fine-tuning
    paddle.jit.save(loaded_layer, "fine-tune.model/linear", input_spec=[x])


此外， ``paddle.jit.save`` 同时保存了模型和参数，如果您只需要从存储结果中载入模型的参数，可以使用 ``paddle.load`` 接口载入，返回所存储模型的state_dict，示例如下：

.. code-block:: python

    import paddle
    import paddle.nn as nn

    IMAGE_SIZE = 784
    CLASS_NUM = 10

    class LinearNet(nn.Layer):
        def __init__(self):
            super(LinearNet, self).__init__()
            self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)

        @paddle.jit.to_static
        def forward(self, x):
            return self._linear(x)

    # create network
    layer = LinearNet()

    # load
    path = "example.model/linear"
    state_dict = paddle.load(path)

    # inference
    layer.set_state_dict(state_dict, use_structured_name=False)
    layer.eval()
    x = paddle.randn([1, IMAGE_SIZE], 'float32')
    pred = layer(x)


四、旧存储格式兼容载入
###################

如果您是从飞桨框架1.x切换到2.x，曾经使用飞桨框架1.x的fluid相关接口存储模型或者参数，飞桨框架2.x也对这种情况进行了兼容性支持，包括以下几种情况。

飞桨1.x模型准备及训练示例，该示例为后续所有示例的前序逻辑：

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

    # enable static mode
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


4.1 从 ``paddle.fluid.io.save_inference_model`` 存储结果中载入模型&参数
------------------------------------------------------------------

(1) 同时载入模型和参数

使用 ``paddle.jit.load`` 配合 ``**configs`` 载入模型和参数。

如果您是按照 ``paddle.fluid.io.save_inference_model`` 的默认格式存储的，可以按照如下方式载入（接前述示例）：

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

如果您指定了存储的模型文件名，可以按照以下方式载入（接前述示例）：

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

如果您指定了存储的参数文件名，可以按照以下方式载入（接前述示例）：

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

(2) 仅载入参数

如果您仅需要从 ``paddle.fluid.io.save_inference_model`` 的存储结果中载入参数，以state_dict的形式配置到已有代码的模型中，可以使用 ``paddle.load`` 配合 ``**configs`` 载入。

如果您是按照 ``paddle.fluid.io.save_inference_model`` 的默认格式存储的，可以按照如下方式载入（接前述示例）：

.. code-block:: python

    model_path = "fc.example.model"

    load_param_dict = paddle.load(model_path)

如果您指定了存储的模型文件名，可以按照以下方式载入（接前述示例）：

.. code-block:: python

    model_path = "fc.example.model.with_model_filename"

    load_param_dict = paddle.load(model_path, model_filename="__simplenet__")

如果您指定了存储的参数文件名，可以按照以下方式载入（接前述示例）：

.. code-block:: python

    model_path = "fc.example.model.with_params_filename"

    load_param_dict = paddle.load(model_path, params_filename="__params__")

.. note::
    一般预测模型不会存储优化器Optimizer的参数，因此此处载入的仅包括模型本身的参数。

.. note::
    由于 ``structured_name`` 是动态图下独有的变量命名方式，因此从静态图存储结果载入的state_dict在配置到动态图的Layer中时，需要配置 ``Layer.set_state_dict(use_structured_name=False)`` 。


4.2 从 ``paddle.fluid.save`` 存储结果中载入参数
----------------------------------------------

 ``paddle.fluid.save`` 的存储格式与2.x动态图接口 ``paddle.save`` 存储格式是类似的，同样存储了dict格式的参数，因此可以直接使用 ``paddle.load`` 载入state_dict，但需要注意不能仅传入保存的路径，而要传入保存参数的文件名，示例如下（接前述示例）：

.. code-block:: python

    # save by fluid.save
    model_path = "fc.example.model.save"
    program = fluid.default_main_program()
    fluid.save(program, model_path)

    # enable dynamic mode
    paddle.disable_static(place)

    load_param_dict = paddle.load("fc.example.model.save.pdparams")


.. note::
    由于 ``paddle.fluid.save`` 接口原先在静态图模式下的定位是存储训练时参数，或者说存储Checkpoint，故尽管其同时存储了模型结构，目前也暂不支持从 ``paddle.fluid.save`` 的存储结果中同时载入模型和参数，后续如有需求再考虑支持。


4.3 从 ``paddle.fluid.io.save_params/save_persistables`` 存储结果中载入参数
-------------------------------------------------------------------------

这两个接口在飞桨1.x版本时，已经不再推荐作为存储模型参数的接口使用，故并未继承至飞桨2.x，之后也不会再推荐使用这两个接口存储参数。

对于使用这两个接口存储参数兼容载入的支持，分为两种情况，下面以 ``paddle.fluid.io.save_params`` 接口为例介绍相关使用方法：

(1) 使用默认方式存储，各参数分散存储为单独的文件，文件名为参数名

这种存储方式仍然可以使用 ``paddle.load`` 接口兼容载入，使用示例如下（接前述示例）：

.. code-block:: python

    # save by fluid.io.save_params
    model_path = "fc.example.model.save_params"
    fluid.io.save_params(exe, model_path)

    # load 
    state_dict = paddle.load(model_path)
    print(state_dict)

(2) 指定了参数存储的文件，将所有参数存储至单个文件中

将所有参数存储至单个文件中会导致存储结果中丢失Tensor名和Tensor数据之间的映射关系，因此这部分丢失的信息需要用户传入进行补足。为了确保正确性，这里不仅要传入Tensor的name列表，同时要传入Tensor的shape和dtype等描述信息，通过检查和存储数据的匹配性确保严格的正确性，这导致载入数据的恢复过程变得比较复杂，仍然需要一些飞桨1.x的概念支持。后续如果此项需求较为普遍，我们将会考虑将该项功能兼容支持到 ``paddle.load`` 中，但由于信息丢失而导致的使用复杂性仍然是存在的，因此建议您避免仅使用这两个接口存储参数。

目前暂时推荐您使用 ``paddle.static.load_program_state`` 接口解决此处的载入问题，需要获取原Program中的参数列表传入该方法，使用示例如下（接前述示例）：

.. code-block:: python

    # save by fluid.io.save_params
    model_path = "fc.example.model.save_params_with_filename"
    fluid.io.save_params(exe, model_path, filename="__params__")

    # load 
    import os
    params_file_path = os.path.join(model_path, "__params__")
    var_list = fluid.default_main_program().all_parameters()
    state_dict = paddle.io.load_program_state(params_file_path, var_list)
