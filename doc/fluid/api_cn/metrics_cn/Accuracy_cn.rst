.. _cn_api_fluid_metrics_Accuracy:

Accuracy
-------------------------------

.. py:class:: paddle.fluid.metrics.Accuracy(name=None)

计算多批次的平均准确率。
https://en.wikipedia.org/wiki/Accuracy_and_precision

参数:
    - **name** — 度量标准的名称

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    # 假设有batch_size = 128
    batch_size=128
    accuracy_manager = fluid.metrics.Accuracy()
    # 假设第一个batch的准确率为0.9
    batch1_acc = 0.9
    accuracy_manager.update(value = batch1_acc, weight = batch_size)
    print("expect accuracy: %.2f, get accuracy: %.2f" % (batch1_acc, accuracy_manager.eval()))
    # 假设第二个batch的准确率为0.8
    batch2_acc = 0.8
    accuracy_manager.update(value = batch2_acc, weight = batch_size)
    #batch1和batch2的联合准确率为(batch1_acc * batch_size + batch2_acc * batch_size) / batch_size / 2
    print("expect accuracy: %.2f, get accuracy: %.2f" % ((batch1_acc * batch_size + batch2_acc * batch_size) / batch_size / 2, accuracy_manager.eval()))
    #重置accuracy_manager
    accuracy_manager.reset()
    #假设第三个batch的准确率为0.8
    batch3_acc = 0.8
    accuracy_manager.update(value = batch3_acc, weight = batch_size)
    print("expect accuracy: %.2f, get accuracy: %.2f" % (batch3_acc, accuracy_manager.eval()))



.. py:method:: update(value, weight)

更新mini batch的状态。

参数：    
    - **value** (float|numpy.array) – 每个mini batch的正确率
    - **weight** (int|float) – batch 大小


.. py:method:: eval()

返回所有累计batches的平均准确率（float或numpy.array）。


