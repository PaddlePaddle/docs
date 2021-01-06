.. _cn_api_distributed_spawn:

spawn
-----

.. py:function:: paddle.distributed.spawn(func, args=(), nprocs=-1, join=True, daemon=False, **options)

使用 ``spawn`` 方法启动多进程任务。

.. note::
    ``spawn``目前仅支持 GPU collective模式.

参数
:::::::::
    - func (function) - 由 ``spawn`` 方法启动的进程所调用的目标函数。该目标函数需要能够被 ``pickled`` (序列化)，所以目标函数必须定义为模块的一级函数，不能是内部子函数或者类方法。
    - args (tuple, 可选) - 传入目标函数 ``func`` 的参数。
    - nprocs (int, 可选) - 启动进程的数目。默认值为-1。当 ``nproc`` 为-1时，模型执行时将会从环境变量中获取当前可用的所有设备进行使用：如果使用GPU执行任务，将会从环境变量 ``CUDA_VISIBLE_DEVICES`` 中获取当前所有可用的设备ID；如果使用CPU执行任务，将会从环境变量 ``CPU_NUM`` 中获取当前可用的CPU设备数，例如，可以通过指令 ``export CPU_NUM=4`` 配置默认可用CPU设备数，如果此环境变量没有设置，将会默认设置该环境变量的值为1。
    - join (bool, 可选) - 对所有启动的进程执行阻塞的 ``join`` ，等待进程执行结束。默认为True。
    - daemon (bool, 可选) - 配置启动进程的 ``daemon`` 属性。默认为False。
    - **options (dict, 可选) - 其他初始化并行执行环境的配置选项。目前支持以下选项： (1) start_method (string) - 启动子进程的方法。进程的启动方法可以是 ``spawn`` ， ``fork`` , ``forkserver`` 。 因为CUDA运行时环境不支持 ``fork`` 方法，当在子进程中使用CUDA时，需要使用 ``spawn`` 或者 ``forkserver`` 方法启动进程。默认方法为 ``spawn`` ； (2) gpus (string) - 指定训练使用的GPU ID, 例如 "0,1,2,3" ， 默认值为None ； (3) ips (string) - 运行集群的节点（机器）IP，例如 "192.168.0.16,192.168.0.17" ，默认值为 "127.0.0.1" 。

返回
:::::::::
 ``MultiprocessContext`` 对象，持有创建的多个进程。

代码示例
:::::::::
.. code-block:: python

    from __future__ import print_function

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

    def train(print_result=False): 
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
        
        if print_result is True:
            print("loss:", loss.numpy())
        
        loss.backward()

        adam.step()
        adam.clear_grad()

    # Usage 1: only pass function. 
    # If your training method no need any argument, and 
    # use all visible devices for parallel training. 
    if __name__ == '__main__':
        dist.spawn(train)

    # Usage 2: pass function and arguments.
    # If your training method need some arguments, and 
    # use all visible devices for parallel training.
    if __name__ == '__main__':
        dist.spawn(train, args=(True,))

    # Usage 3: pass function, arguments and nprocs.
    # If your training method need some arguments, and 
    # only use part of visible devices for parallel training.
    # If your machine hold 8 cards {0,1,2,3,4,5,6,7},
    # this case will use cards {0,1}; If you set 
    # CUDA_VISIBLE_DEVICES=4,5,6,7, this case will use
    # cards {4,5}
    if __name__ == '__main__':
        dist.spawn(train, args=(True,), nprocs=2)

    # Usage 4: pass function, arguments, nprocs and selected_gpus.
    # If your training method need some arguments, and 
    # only use part of visible devices for parallel training,
    # but you can't set your machine's environment variable 
    # CUDA_VISIBLE_DEVICES, such as it is None or all cards
    # {0,1,2,3,4,5,6,7}, you can pass `gpus` to 
    # select the GPU cards you want to use. For example,
    # this case will use cards {4,5} if your machine hold 8 cards.
    if __name__ == '__main__':
        dist.spawn(train, args=(True,), nprocs=2, gpus='4,5')