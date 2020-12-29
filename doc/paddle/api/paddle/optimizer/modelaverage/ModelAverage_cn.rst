.. _cn_api_paddle_optimizer_ModelAverage:

ModelAverage
-------------------------------


.. py:class:: paddle.optimizer.ModelAverage(inner_optimizer, average_window_rate, parameters=None, min_average_window=10000, max_average_window=10000, name=None)


ModelAverage优化器，在训练过程中累积特定连续的历史Parameters，累积的历史范围可以用传入的average_window参数来控制，在预测时使用平均后的Parameters，通常可以提高预测的精度。

在滑动窗口中累积Parameters的平均值，将结果将保存在临时变量中，通过调用 ``apply()`` 方法可应用于当前模型的Parameters，使用 ``restore()`` 方法恢复当前模型Parameters的值。

计算平均值的窗口大小由 ``average_window_rate`` ， ``min_average_window`` ， ``max_average_window`` 以及当前Parameters更新次数(num_updates)共同决定。

累积次数（num_accumulates）大于特定窗口阈值(average_window)时，将累积的Parameters临时变量置为0.0，这几个参数的作用通过以下示例代码说明：

.. code-block:: python

    if num_accumulates >= min_average_window and num_accumulates >= min(max_average_window, num_updates * average_window_rate):
        num_accumulates = 0

上述条件判断语句中，num_accumulates表示当前累积的次数，可以抽象理解为累积窗口的长度，窗口长度至少要达到min_average_window参数设定的长度，并且不能超过max_average_window参数或者num_updates * average_window_rate规定的长度，其中num_updates表示当前Parameters更新的次数，average_window_rate是一个计算窗口长度的系数。
 
参数：
  - **inner_optimizer** (Optimizer) - 基础优化器，如SGD, Momentum等
  - **average_window_rate** (float) – 相对于Parameters更新次数的窗口长度计算比率
  - **parameters** (list, 可选) - 指定优化器需要优化的参数。默认值为None，这时若基础优化器（ `inner_optimize` )的待优化参数不为空，则设置为基础优化器的所有待优化参数，反之则所有的 `do_model_average` 属性为True的参数都将被优化。
  - **min_average_window** (int, 可选) – 平均值计算窗口长度的最小值，默认值为10000
  - **max_average_window** (int, 可选) – 平均值计算窗口长度的最大值，推荐设置为一轮训练中mini-batchs的数目，默认值为10000
  - **name** (str, 可选)– 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None

**代码示例**

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
            self.bias = self._linear.bias

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
                print("Train Epoch {} batch {}: loss = {}, bias = {}".format(
                    epoch_id, batch_id, np.mean(loss.numpy()), layer.bias.numpy()))
    def evaluate(layer, loader, loss_fn):
        for batch_id, (image, label) in enumerate(loader()):
            out = layer(image)
            loss = loss_fn(out, label)
            loss.backward()
            print("Evaluate batch {}: loss = {}, bias = {}".format(
                batch_id, np.mean(loss.numpy()), layer.bias.numpy()))

    # create network
    layer = LinearNet()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = opt.Momentum(learning_rate=0.2, momentum=0.1, parameters=layer.parameters())
    # build ModelAverage optimizer
    model_average = paddle.optimizer.ModelAverage(optimizer, 0.15,
                                                min_average_window=2,
                                                max_average_window=10)

    # create data loader
    dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
    loader = paddle.io.DataLoader(dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=2)
    # create data loader
    eval_loader = paddle.io.DataLoader(dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=1)

    # train
    train(layer, loader, loss_fn, model_average)

    print("\nEvaluate With ModelAverage")
    with model_average.apply(need_restore=False):
        evaluate(layer, eval_loader, loss_fn)

    print("\nEvaluate With Restored Paramters")
    model_average.restore()
    evaluate(layer, eval_loader, loss_fn) 

.. py:method:: minimize(loss, startup_program=None, parameters=None, no_grad_set=None)

为网络添加反向计算过程，并根据反向计算所得的梯度，更新parameters中的Parameters，最小化网络损失值loss。

参数：
    - **loss** (Tensor) – 需要最小化的损失值变量
    - **startup_program** (Program, 可选) – 用于初始化parameters中参数的 :ref:`cn_api_fluid_Program` , 默认值为None，此时将使用 :ref:`cn_api_fluid_default_startup_program` 
    - **parameters** (list, 可选) – 待更新的Parameter或者Parameter.name组成的列表， 默认值为None，此时将更新所有的Parameter
    - **no_grad_set** (set, 可选) – 不需要更新的Parameter或者Parameter.name组成的集合，默认值为None
         
返回: tuple(optimize_ops, params_grads)，其中optimize_ops为参数优化OP列表；param_grads为由(param, param_grad)组成的列表，其中param和param_grad分别为参数和参数的梯度。在静态图模式下，该返回值可以加入到 ``Executor.run()`` 接口的 ``fetch_list`` 参数中，若加入，则会重写 ``use_prune`` 参数为True，并根据 ``feed`` 和 ``fetch_list`` 进行剪枝，详见 ``Executor`` 的文档。


**代码示例**

.. code-block:: python

    import paddle
    import numpy as np
    inp = paddle.ones(shape=[1, 10], dtype='float32')
    linear = paddle.nn.Linear(10, 1)
    out = linear(inp)
    loss = paddle.mean(out)
    sgd = paddle.optimizer.SGD(learning_rate=0.1,parameters=linear.parameters())

    modelaverage = paddle.optimizer.ModelAverage(sgd,
                                                0.15,
                                                parameters=linear.parameters(),
                                                min_average_window=2,
                                                max_average_window=4)
    loss.backward()
    modelaverage.minimize(loss)
    modelaverage.clear_grad()

.. py:method:: step()

**注意：**

  **1. 该API只在** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**

执行一次优化器并进行参数更新。

返回：None。


**代码示例**

.. code-block:: python

    import paddle
    import numpy as np
    inp = paddle.ones(shape=[1, 10], dtype='float32')
    linear = paddle.nn.Linear(10, 1)
    out = linear(inp)
    loss = paddle.mean(out)
    sgd = paddle.optimizer.SGD(learning_rate=0.1,parameters=linear.parameters())

    modelaverage = paddle.optimizer.ModelAverage(sgd,
                                                0.15,
                                                parameters=linear.parameters(),
                                                min_average_window=2,
                                                max_average_window=4)
    loss.backward()
    modelaverage.step()
    modelaverage.clear_grad()

.. py:method:: apply(executor=None, need_restore=True)

将累积Parameters的平均值应用于当前网络的Parameters。

参数：
    - **executor** (Executor|None) – 静态图模式下设置为当前网络的执行器，动态图模式下设置为None
    - **need_restore** (bool) – 恢复标志变量，设为True时，执行完成后会将网络的Parameters恢复为网络默认的值，设为False将不会恢复，默认值True

返回：无

**代码示例**

.. code-block:: python
        
    import paddle
    import numpy as np
    inp = paddle.ones(shape=[1, 10], dtype='float32')
    linear = paddle.nn.Linear(10, 1)
    out = linear(inp)
    loss = paddle.mean(out)
    sgd = paddle.optimizer.SGD(learning_rate=0.1,parameters=linear.parameters())

    modelaverage = paddle.optimizer.ModelAverage(sgd,
                                                0.15,
                                                parameters=linear.parameters(),
                                                min_average_window=2,
                                                max_average_window=4)
    loss.backward()
    modelaverage.step()
    
    with modelaverage.apply():
        for param in linear.parameters():
            print(param)

    for param in linear.parameters():
        print(param)

.. py:method:: restore(executor=None)

恢复当前网络的Parameters值

参数：
    - **executor** (Executor|None) – 静态图模式下设置为当前网络的执行器，动态图模式下设置为None

返回：无

**代码示例**

.. code-block:: python
        
    import paddle
    import numpy as np
    inp = paddle.ones(shape=[1, 10], dtype='float32')
    linear = paddle.nn.Linear(10, 1)
    out = linear(inp)
    loss = paddle.mean(out)
    sgd = paddle.optimizer.SGD(learning_rate=0.1,parameters=linear.parameters())

    modelaverage = paddle.optimizer.ModelAverage(sgd,
                                                0.15,
                                                parameters=linear.parameters(),
                                                min_average_window=2,
                                                max_average_window=4)
    loss.backward()
    modelaverage.step()
    
    with modelaverage.apply(need_restore=False):
        for param in linear.parameters():
            print(param)

    for param in linear.parameters():
        print(param)

    modelaverage.restore()

    for param in linear.parameters():
        print(param)