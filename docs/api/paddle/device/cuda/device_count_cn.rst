.. _cn_api_device_cuda_device_count:

device_count
-------------------------------

.. py:function:: paddle.device.cuda.device_count()

返回值是int，表示当前程序可用的GPU数量。

返回： 返回一个整数，表示当前程序可用的GPU数量。


**代码示例**：

.. code-block:: python
           
    import paddle
    
    paddle.device.cuda.device_count()
