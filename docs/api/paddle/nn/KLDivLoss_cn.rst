.. _cn_api_paddle_nn_KLDivLoss:

KLDivLoss
-------------------------------

.. py:class:: paddle.nn.KLDivLoss(reduction='mean')

创建一个 `KLDivLoss` 类的可调用对象，以计算输入(Input)和输入(Label)之间的 Kullback-Leibler 散度损失。注意其中输入(Input)应为对数概率值，输入(Label)应为概率值。

kL 发散损失计算如下：

..  math::

    l(input, label) = label * (log(label) - input)


当 ``reduction``  为 ``none`` 时，输出损失与输入（input）形状相同，各点的损失单独计算，不会对结果做 reduction 。

当 ``reduction``  为 ``mean`` 时，输出损失的形状为[]，输出为所有损失的平均值。

当 ``reduction``  为 ``sum`` 时，输出损失的形状为[]，输出为所有损失的总和。

当 ``reduction``  为 ``batchmean`` 时，输出损失为[N]的形状，N 为批大小，输出为所有损失的总和除以批量大小。

参数
::::::::::::

    - **reduction** (str，可选) - 要应用于输出的 reduction 类型，可用类型为‘none’ | ‘batchmean’ | ‘mean’ | ‘sum’，‘none’表示无 reduction，‘batchmean’ 表示输出的总和除以批大小，‘mean’ 表示所有输出的平均值，‘sum’表示输出的总和。

形状
::::::::::::

    - **input** (Tensor)：输入的 Tensor，维度是[N, *]，其中 N 是 batch size， `*` 是任意数量的额外维度。数据类型为：float32、float64。
    - **label** (Tensor)：标签，维度是[N, *]，与 ``input`` 相同。数据类型为：float32、float64。
    - **output** (Tensor)：输入 ``input`` 和标签 ``label`` 间的 kl 散度。如果 `reduction` 是 ``'none'``，则输出 Loss 的维度为 [N, *]，与输入 ``input`` 相同。如果 `reduction` 是 ``'batchmean'`` 、 ``'mean'`` 或 ``'sum'``，则输出 Loss 的维度为 []。

代码示例
::::::::::::

COPY-FROM: paddle.nn.KLDivLoss
