.. _cn_api_fluid_dygraph_DataParallel:

DataParallel
------------

.. py:class:: paddle.fluid.dygraph.DataParallel(layers, strategy)


通过数据并行模式执行动态图模型。

目前，``DataParallel`` 仅支持以多进程的方式执行动态图模型。

支持两种使用方式：

1. 使用 ``paddle.distributed.spawn`` 方法启动，例如：

 ``python demo.py`` (spawn need to be called in ``__main__`` method)

2. 使用 ``paddle.distributed.launch`` 方法启动，例如：

``python -m paddle.distributed.launch –selected_gpus=0,1 demo.py``

其中 ``demo.py`` 脚本的代码可以是下面的示例代码。

参数：
    - **Layer** (Layer) - 需要通过数据并行方式执行的模型。
    - **strategy** (ParallelStrategy，可选) - (deprecated) 数据并行的策略，包括并行执行的环境配置。默认为None。

返回：支持数据并行的 ``Layer``

返回类型：Layer实例

**代码示例**：

.. code-block:: python

    import paddle
    import paddle.nn as nn
    import paddle.optimizer as opt
    import paddle.distributed as dist

    class LinearNet(nn.Layer):
        def __init__(self):
            super(LinearNet, self).__init__()
            self._linear1 = nn.Linear(10, 10)
            self._linear2 = nn.Linear(10, 1)
            
        def forward(self, x):
            return self._linear2(self._linear1(x))

    def train():
        # 1. enable dynamic mode
        paddle.disable_static()
        
        # 2. initialize parallel environment
        dist.init_parallel_env()

        # 3. create data parallel layer & optimizer
        layer = LinearNet()
        dp_layer = paddle.DataParallel(layer)

        loss_fn = nn.MSELoss()
        adam = opt.Adam(
            learning_rate=0.001, parameters=dp_layer.parameters())

        # 4. run layer
        inputs = paddle.randn([10, 10], 'float32')
        outputs = dp_layer(inputs)
        labels = paddle.randn([10, 1], 'float32')
        loss = loss_fn(outputs, labels)
        
        loss.backward()

        adam.step()
        adam.clear_grad()

    if __name__ == '__main__':
        # 1. start by ``paddle.distributed.spawn`` (default)
        dist.spawn(train, nprocs=2)
        # 2. start by ``paddle.distributed.launch``
        # train()
