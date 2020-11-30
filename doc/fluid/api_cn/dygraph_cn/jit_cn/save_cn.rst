.. _cn_api_fluid_dygraph_jit_save:

save
-----------------

.. py:function:: paddle.fluid.dygraph.jit.save(layer, model_path, input_spec=None, configs=None)

将输入的经过 ``@declarative`` 装饰的 :ref:`cn_api_fluid_dygraph_Layer` 存储为 :ref:`cn_api_fluid_dygraph_TranslatedLayer` 格式的模型，
载入后可用于预测推理或者fine-tune训练。

该接口将会将输入 :ref:`cn_api_fluid_dygraph_Layer` 转写后的模型结构 ``Program`` 和所有必要的持久参数变量存储至输入路径 ``model_path`` 中。

默认存储的 ``Program`` 文件名为 ``__model__``， 默认存储持久参数变量的文件名为 ``__variables__``，
同时会将变量的一些描述信息存储至文件 ``__variables.info__``，这些额外的信息将在fine-tune训练中使用。

存储的模型能够被以下API载入使用：
  - :ref:`cn_api_fluid_dygraph_jit_load`
  - :ref:`cn_api_fluid_io_load_inference_model` （需要配置参数 ``params_filename='__variables__'`` ）
  - 其他预测库API

参数：
    - **layer** (Layer) - 需要存储的 :ref:`cn_api_fluid_dygraph_Layer` 对象。输入的 ``Layer`` 需要经过 ``@declarative`` 装饰。
    - **model_path** (str) - 存储模型的目录。
    - **input_spec** (list[Variable], 可选) - 描述存储模型的输入。此参数是传入当前存储的 ``TranslatedLayer`` forward方法的一个示例输入。如果为 ``None`` ，所有原 ``Layer`` forward方法的输入变量将都会被配置为存储模型的输入变量。默认为 ``None``。
    - **configs** (SaveLoadConfig, 可选) - 用于指定额外配置选项的 :ref:`cn_api_fluid_dygraph_jit_SaveLoadConfig` 对象。默认为 ``None``。

返回：无

**示例代码**

.. code-block:: python

    import numpy as np
    import paddle.fluid as fluid
    from paddle.fluid.dygraph import Linear
    from paddle.fluid.dygraph import declarative
    BATCH_SIZE = 32
    BATCH_NUM = 20
    def random_batch_reader():
        def _get_random_images_and_labels(image_shape, label_shape):
            image = np.random.random(size=image_shape).astype('float32')
            label = np.random.random(size=label_shape).astype('int64')
            return image, label
        def __reader__():
            for _ in range(BATCH_NUM):
                batch_image, batch_label = _get_random_images_and_labels(
                    [BATCH_SIZE, 784], [BATCH_SIZE, 1])
                yield batch_image, batch_label
        return __reader__
    class LinearNet(fluid.dygraph.Layer):
        def __init__(self, in_size, out_size):
            super(LinearNet, self).__init__()
            self._linear = Linear(in_size, out_size)
        @declarative
        def forward(self, x):
            return self._linear(x)
    # 开启命令式编程模式
    fluid.enable_dygraph() 
    # 创建网络
    net = LinearNet(784, 1)
    adam = fluid.optimizer.AdamOptimizer(learning_rate=0.1, parameter_list=net.parameters())
    # 创建DataLoader
    train_loader = fluid.io.DataLoader.from_generator(capacity=5)
    train_loader.set_batch_generator(random_batch_reader())
    # 训练
    for data in train_loader():
        img, label = data
        label.stop_gradient = True
        cost = net(img)
        loss = fluid.layers.cross_entropy(cost, label)
        avg_loss = fluid.layers.mean(loss)
        avg_loss.backward()
        adam.minimize(avg_loss)
        net.clear_gradients()
    # 存储模型
    model_path = "linear.example.model"
    fluid.dygraph.jit.save(
        layer=net,
        model_path=model_path,
        input_spec=[img])
