.. _cn_api_distributed_init_parallel_env:

init_parallel_env
-----------------

.. py:function:: paddle.distributed.init_parallel_env()

初始化动态图模式下的并行训练环境。

.. note::
    目前同时初始化 ``NCCL`` 和 ``GLOO`` 上下文用于通信。

返回
:::::::::
无

代码示例
:::::::::
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
        # 1. initialize parallel environment
        dist.init_parallel_env()

        # 2. create data parallel layer & optimizer
        layer = LinearNet()
        dp_layer = paddle.DataParallel(layer)

        loss_fn = nn.MSELoss()
        adam = opt.Adam(
            learning_rate=0.001, parameters=dp_layer.parameters())

        # 3. run layer
        inputs = paddle.randn([10, 10], 'float32')
        outputs = dp_layer(inputs)
        labels = paddle.randn([10, 1], 'float32')
        loss = loss_fn(outputs, labels)
        
        loss.backward()

        adam.step()
        adam.clear_grad()

    if __name__ == '__main__':
        dist.spawn(train)
