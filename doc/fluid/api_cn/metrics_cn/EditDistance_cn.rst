.. _cn_api_fluid_metrics_EditDistance:

EditDistance
-------------------------------

.. py:class:: paddle.fluid.metrics.EditDistance(name)

用于管理字符串的编辑距离。编辑距离是通过计算将一个字符串转换为另一个字符串所需的最小编辑操作数（添加、删除或替换）来量化两个字符串（例如单词）彼此不相似的程度一种方法。 参考 https://en.wikipedia.org/wiki/Edit_distance。

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid
    import numpy as np
    
    # 假设batch_size为128
    batch_size = 128
    
    # 初始化编辑距离管理器
    distance_evaluator = fluid.metrics.EditDistance("EditDistance")
    # 生成128个序列对间的编辑距离，此处的最大距离是10
    edit_distances_batch0 = np.random.randint(low = 0, high = 10, size = (batch_size, 1))
    seq_num_batch0 = batch_size

    distance_evaluator.update(edit_distances_batch0, seq_num_batch0)
    avg_distance, wrong_instance_ratio = distance_evaluator.eval()
    print("the average edit distance for batch0 is %.2f and the wrong instance ratio is %.2f " % (avg_distance, wrong_instance_ratio))
    edit_distances_batch1 = np.random.randint(low = 0, high = 10, size = (batch_size, 1))
    seq_num_batch1 = batch_size

    distance_evaluator.update(edit_distances_batch1, seq_num_batch1)
    avg_distance, wrong_instance_ratio = distance_evaluator.eval()
    print("the average edit distance for batch0 and batch1 is %.2f and the wrong instance ratio is %.2f " % (avg_distance, wrong_instance_ratio))


.. py:method:: reset()

清空存储结果。

参数：无

返回：无


.. py:method:: update(distances, seq_num)

更新存储结果

参数：
    - **distances** – 一个形状为(batch_size, 1)的numpy.array，每个元素代表两个序列间的距离。
    - **seq_num** – 一个整型/浮点型值，代表序列对的数量。

返回：无

.. py:method:: eval()

返回两个浮点数：
avg_distance：使用更新函数更新的所有序列对的平均距离。
avg_instance_error：编辑距离不为零的序列对的比例。





