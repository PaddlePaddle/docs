..  _api_guide_executor:

##########
执行引擎
##########

:code:`Executor` 实现了一个简易的执行器，所有的操作在其中顺序执行。你可以在 Python 脚本中运行 :code:`Executor` 。

 :code:`Executor` 的逻辑非常简单。建议在调试阶段用 :code:`Executor` 在一台计算机上完整地运行模型，然后转向多设备或多台计算机计算。

 :code:`Executor` 在构造时接受一个 :code:`PlaceLike` ，它既可能是具体的 :code:`Place` 对象，包括 :code:`CPUPlace` :code:`CUDAPlace` 等，也可能是一个 :code:`str` 类型的设备描述符，如 :code:`"cpu"` 或 :code:`"gpu:0"`。可参照 :code:`PlaceLike` 的 `源代码 <https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/_typing/device_like.py#L39>`_ 定义

.. code-block:: python

    import paddle
    # 首先创建 Executor。
    paddle.enable_static()
    place = paddle.CUDAPlace(0) if 'gpu:0' in paddle.device.get_available_device() else paddle.CPUPlace()
    exe = paddle.static.Executor(place)

    # 运行启动程序仅一次。
    exe.run(paddle.static.default_startup_program())

简单样例请参照 `代码示例 <../../api/paddle/static/Executor_cn.html#daimashili>`_

- 相关 API :

 - :ref:`cn_api_paddle_static_Executor`
