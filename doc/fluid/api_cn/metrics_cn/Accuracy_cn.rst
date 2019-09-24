.. _cn_api_fluid_metrics_Accuracy:

Accuracy
-------------------------------
.. py:class:: paddle.fluid.metrics.Accuracy(name=None)

该接口用来计算多个mini-batch的平均准确率。Accuracy对象有两个状态value和weight。Accuracy的定义参照 https://en.wikipedia.org/wiki/Accuracy_and_precision 。

参数:
    - **name** (None|str) – 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None。

返回：初始化后的 ``Accuracy`` 对象

返回类型：Accuracy

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

该函数使用输入的(value, weight)来累计更新Accuracy对象的对应状态，更新方式如下：

    .. math::
                   \\ \begin{array}{l}{\text { self. value }+=\text { value } * \text { weight }} \\ {\text { self. weight }+=\text { weight }}\end{array} \\

参数：    
    - **value** (float|numpy.array) – mini-batch的正确率
    - **weight** (int|float) – mini-batch的大小

返回：无

.. py:method:: eval()

该函数计算并返回累计的mini-batches的平均准确率。

返回：累计的mini-batches的平均准确率

返回类型：float或numpy.array

