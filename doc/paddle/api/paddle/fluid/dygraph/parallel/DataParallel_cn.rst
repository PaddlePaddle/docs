.. _cn_api_fluid_dygraph_DataParallel:

DataParallel
------------

.. py:class:: paddle.DataParallel(layers, strategy=None, comm_buffer_size=25, last_comm_buffer_size=1)


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
    - **comm_buffer_size** (int，可选) - 它是通信调用（如NCCLAllReduce）时，参数梯度聚合为一组的内存大小（MB）。默认值：25。
    - **last_comm_buffer_size** （float，可选）它限制通信调用中最后一个缓冲区的内存大小（MB）。减小最后一个通信缓冲区的大小有助于提高性能。默认值：1。默认值：1    

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
        # 1. start by ``paddle.distributed.spawn`` (default)
        dist.spawn(train, nprocs=2)
        # 2. start by ``paddle.distributed.launch``
        # train()

.. py:method:: state_dict(destination=None, include_sublayers=True)

获取当前层及其子层的所有parameters和持久的buffers。并将所有parameters和buffers存放在dict结构中。

参数：
    - **destination** (dict, 可选) - 如果提供 ``destination`` ，则所有参数和持久的buffers都将存放在 ``destination`` 中。 默认值：None。
    - **include_sublayers** (bool, 可选) - 如果设置为True，则包括子层的参数和buffers。默认值：True。

返回：dict， 包含所有parameters和持久的buffers的dict

**代码示例**

.. code-block:: python

    import paddle
    import paddle.distributed as dist

    dist.init_parallel_env()

    emb = fluid.dygraph.Embedding([10, 10])
    emb = fluid.dygraph.DataParallel(emb)

    state_dict = emb.state_dict()
    paddle.save(state_dict, "paddle_dy.pdparams")

.. py:method:: set_state_dict(state_dict, use_structured_name=True)

根据传入的 ``state_dict`` 设置parameters和持久的buffers。 所有parameters和buffers将由 ``state_dict`` 中的 ``Tensor`` 设置。

参数：
    - **state_dict** (dict) - 包含所有parameters和可持久性buffers的dict。
    - **use_structured_name** (bool, 可选) - 如果设置为True，将使用Layer的结构性变量名作为dict的key，否则将使用Parameter或者Buffer的变量名作为key。默认值：True。

返回：无

**代码示例**

.. code-block:: python

    import paddle
    import paddle.distributed as dist

    dist.init_parallel_env()

    emb = paddle.nn.Embedding(10, 10)
    emb = fluid.dygraph.DataParallel(emb)

    state_dict = emb.state_dict()
    paddle.save(state_dict, "paddle_dy.pdparams")

    para_state_dict = paddle.load("paddle_dy.pdparams")
    emb.set_state_dict(para_state_dict)
