.. _cn_api_incubate_LookAhead:

LookAhead
-------------------------------

.. py:function:: class paddle.incubate.LookAhead(inner_optimizer, alpha=0.5, k=5, name=None)
此 API 为论文 `Lookahead Optimizer: k steps forward, 1 step back <https://arxiv.org/abs/1907.08610>`_ 中 Lookahead 优化器的实现。
Lookahead 保留两组参数：fast_params 和 slow_params。每次训练迭代中 inner_optimizer 更新 fast_params。
Lookahead 每 k 次训练迭代更新 slow_params 和 fast_params，如下所示：

.. math::

        slow\_param_t & = slow\_param_{t-1} + alpha * (fast\_param_{t-1} - slow\_param_{t-1})

        fast\_param_t & = slow\_param_t


参数
:::::::::
    - **inner_optimizer** (inner_optimizer) - 每次迭代更新 fast params 的优化器。
    - **alpha** (float，可选) - Lookahead 的学习率。默认值为 0.5。
    - **k** (int，可选) - slow params 每 k 次迭代更新一次。默认值为 5。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

代码示例
:::::::::

        .. code-block:: python

            import numpy as np
            import paddle
            import paddle.nn as nn

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
                    label = np.random.randint(0, CLASS_NUM - 1,
                                            (1, )).astype('int64')
                    return image, label

                def __len__(self):
                    return self.num_samples

            class LinearNet(nn.Layer):
                def __init__(self):
                    super().__init__()
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
                        print("Train Epoch {} batch {}: loss = {}".format(
                            epoch_id, batch_id, np.mean(loss.numpy())))

            layer = LinearNet()
            loss_fn = nn.CrossEntropyLoss()
            optimizer = paddle.optimizer.SGD(learning_rate=0.1, parameters=layer.parameters())
            lookahead = paddle.incubate.LookAhead(optimizer, alpha=0.2, k=5)

            # create data loader
            dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
            loader = paddle.io.DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                drop_last=True,
                num_workers=2)

            train(layer, loader, loss_fn, lookahead)

方法
:::::::::


step()
'''''''''

执行优化器并更新参数一次。

**返回**

None。


**代码示例**

            .. code-block:: python

                import paddle
                import numpy as np

                inp = paddle.to_tensor(np.random.random([1, 10]).astype('float32'))
                linear = paddle.nn.Linear(10, 1)
                out = linear(inp)
                loss = paddle.mean(out)
                sgd = paddle.optimizer.SGD(learning_rate=0.1,parameters=linear.parameters())
                lookahead = paddle.incubate.LookAhead(sgd, alpha=0.2, k=5)
                loss.backward()
                lookahead.step()
                lookahead.clear_grad()

minimize(loss, startup_program=None, parameters=None, no_grad_set=None)
'''''''''

增加操作以通过更新参数来最小化损失。

**参数**

    - **loss** (Tensor) - 包含要最小化的值的 Tensor。
    - **startup_program** (Program，可选) - :ref:`cn_api_fluid_Program`。在 ``parameters`` 中初始化参数。默认值为 None，此时将使用 :ref:`cn_api_fluid_default_startup_program` 。
    - **parameters** (list，可选) - 列出更新最小化 ``loss`` 的 ``Tensor`` 或 ``Tensor.name``。默认值为 None，此时所有参数都会被更新。
    - **no_grad_set** (set，可选) - 不需要更新的 ``Tensor`` 或 ``Tensor.name`` 的集合。默认值为 None。

**返回**

tuple: tuple (optimize_ops, params_grads)，由 ``minimize`` 添加的操作列表和 ``(param, grad)`` Tensor 对的列表，其中 param 是参数，grad 参数对应的梯度值。在静态图模式中，返回的元组可以传给 ``Executor.run()`` 中的 ``fetch_list`` 来表示程序剪枝。这样程序在运行之前会通过 ``feed`` 和 ``fetch_list`` 被剪枝，详情请参考 ``Executor`` 。

**代码示例**

            .. code-block:: python

                import paddle
                import numpy as np

                inp = paddle.to_tensor(np.random.random([1, 10]).astype('float32'))
                linear = paddle.nn.Linear(10, 1)
                out = linear(inp)
                loss = paddle.mean(out)
                sgd = paddle.optimizer.SGD(learning_rate=0.1,parameters=linear.parameters())
                lookahead = paddle.incubate.LookAhead(sgd, alpha=0.2, k=5)
                loss.backward()
                lookahead.minimize(loss)
                lookahead.clear_grad()
