.. _cn_api_paddle_static_auc:

auc
-------------------------------

.. py:function:: paddle.static.auc(input, label, curve='ROC', num_thresholds=4095, topk=1, slide_steps=1, ins_tag_weight=None)




**Area Under the Curve(AUC) Layer**

该层根据前向输出和标签计算 AUC，在二分类(binary classification)估计中广泛使用。

注：如果输入标注包含一种值，只有 0 或 1 两种情况，数据类型则强制转换成布尔值。

相关定义可以在这里找到：https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve

有两种可能的曲线：

1. ROC：受试者工作特征曲线

2. PR：准确率召回率曲线

参数
::::::::::::

    - **input** (Tensor) - 数据类型为 float32、float64。浮点二维变量，值的范围为[0,1]。每一行降序排列。该输入为网络预测值，通常代表每个标签的概率。
    - **label** (Tensor) - 数据类型为 int32、int64。二维整型变量，为训练数据的标签，第一维大小代表 batch size，第二维大小为 1。
    - **curve** (str，可选) - 曲线类型，可以为 ``ROC`` 或 ``PR``，默认 ``ROC``。
    - **num_thresholds** (int，可选) - 将 roc 曲线离散化时使用的临界值数。默认 4095。
    - **topk** (int，可选) -  取 topk 的输出值用于计算。
    - **slide_steps** (int，可选) - 当计算 batch auc 时，不仅用当前步也用于先前步。slide_steps=1，表示用当前步；slide_steps = 3 表示用当前步和前两步；slide_steps = 0，则用所有步。默认值为 1。
    - **ins_tag_weight** (Tensor，可选) - 在多 instag 场景下，该数值代表着数据的真伪性，如果为 0，说明数据是被填充的假数据，如果为 1，说明为真数据。默认为 None，此时该数值被赋值为 1。

返回
::::::::::::
tuple，当前计算出的 AUC。数据类型是 Tensor，支持 float32 和 float64。

返回的元组为 auc_out, batch_auc_out, [batch_stat_pos, batch_stat_neg, stat_pos, stat_neg]。

- auc_out 为准确率的结果；
- batch_auc_out 为 batch 准确率的结果；
- batch_stat_pos 为 batch 计算时 label=1 的统计值；
- batch_stat_neg 为 batch 计算时 label=0 的统计值；
- stat_pos 计算时 label=1 的统计值；
- stat_neg 为计算时 label=0 的统计值。

代码示例
::::::::::::

COPY-FROM: paddle.static.auc
