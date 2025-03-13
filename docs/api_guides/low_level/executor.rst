..  _api_guide_executor:

##########
执行器
##########

:code:`Executor` 实现了一个简易的执行器，所有的操作在其中顺序执行。你可以在 Python 脚本中运行 :code:`Executor` 。

 :code:`Executor` 的逻辑非常简单。建议在调试阶段用 :code:`Executor` 在一台计算机上完整地运行模型，然后转向多设备或多台计算机计算。

 :code:`Executor` 在构造时接受一个 :code:`Place` 对象，包括 :ref:`cn_api_paddle_CPUPlace` 与 :ref:`cn_api_paddle_CUDAPlace` 等，也可以是一个 :code:`str` 类型的设备描述符，如 :code:`"cpu"` 或 :code:`"gpu:0"`。

.. code-block:: python

    import paddle
    import numpy as np

    # 开启静态图模式，设置计算设备 place
    paddle.enable_static()
    place = (
        paddle.CUDAPlace(0)
        if "gpu:0" in paddle.device.get_available_device()
        else paddle.CPUPlace()
    )

    # 首先创建 Executor，用于执行计算图
    exe = paddle.static.Executor(place)

    # 定义输入数据和标签，shape 中的 -1 表示 batch size 是可变的，784 是特征维度
    x = paddle.static.data(name="x", shape=[-1, 784], dtype="float32")
    label = paddle.static.data(name="label", shape=[-1, 1], dtype="int64")

    # 定义输出维度为 10 的全连接层，损失函数
    out = paddle.static.nn.fc(name="fc", x=x, size=10, activation="relu")
    loss = paddle.nn.functional.cross_entropy(out, label)

    # 运行启动程序 startup_program 初始化参数
    exe.run(paddle.static.default_startup_program())

    # 准备输入数据，创建一个 1x784 的全 1 矩阵作为输入数据，一个 1x1 的 0 作为标签
    feed_dict = {
        "x": np.ones((1, 784), dtype="float32"),
        "label": np.array([[0]], dtype="int64"),
    }

    # 执行主程序，将 feed_dict 作为输入，获取 loss 和 out 变量的值
    (loss_value, out_value) = exe.run(
        paddle.static.default_main_program(), feed=feed_dict, fetch_list=[loss, out]
    )
    print(f"loss: {loss_value}\nmodel output: {out_value}")

关于执行器的参数介绍，可以参考 :ref:`cn_api_paddle_static_Executor` 这部分内容。
