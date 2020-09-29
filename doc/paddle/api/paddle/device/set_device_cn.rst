.. _cn_api_set_device:

set_device
-------------------------------

.. py:function:: paddle.set_device(device)


Paddle支持包括CPU和GPU在内的多种设备运行，设备可以通过字符串标识符表示，此功能可以指定OP运行的全局设备。

参数：
    - **device** (str)- 此参数确定特定的运行设备，它可以是 ``cpu`` 或者 ``gpu:0`` ，当 ``device`` 是 ``cpu`` 的时候， 程序在CPU上运行， 当device是 ``gpu`` 的时候，程序在GPU上运行。


**代码示例**

.. code-block:: python
    
    import paddle
    
    paddle.set_device("cpu")
    x1 = paddle.ones(name='x1', shape=[1, 2], dtype='int32')
    x2 = paddle.zeros(name='x2', shape=[1, 2], dtype='int32')
    data = paddle.stack([x1,x2], axis=1)
    
