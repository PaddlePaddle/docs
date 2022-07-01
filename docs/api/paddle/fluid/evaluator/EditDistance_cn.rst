.. _cn_api_fluid_metrics_EditDistance:

EditDistance
-------------------------------

.. py:class:: paddle.fluid.metrics.EditDistance(name)




用于管理字符串的编辑距离。编辑距离是通过计算将一个字符串转换为另一个字符串所需的最小编辑操作数（添加、删除或替换）来量化两个字符串（例如单词）彼此不相似的程度一种方法。参考 https://en.wikipedia.org/wiki/Edit_distance。

代码示例
::::::::::::


COPY-FROM: paddle.fluid.metrics.EditDistance

方法
::::::::::::
reset()
'''''''''

清空存储结果。

**参数**
无

**返回**
无


update(distances, seq_num)
'''''''''

更新存储结果

**参数**

    - **distances** – 一个形状为(batch_size, 1)的numpy.array，每个元素代表两个序列间的距离。
    - **seq_num** – 一个整型/浮点型值，代表序列对的数量。

**返回**
无

eval()
'''''''''

返回两个浮点数：
avg_distance：使用更新函数更新的所有序列对的平均距离。
avg_instance_error：编辑距离不为零的序列对的比例。





