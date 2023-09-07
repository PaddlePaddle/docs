.. _cn_api_paddle_static_nn_sequence_softmax:

sequence_softmax
-------------------------------


.. py:function:: paddle.static.nn.sequence_softmax(input, use_cudnn=False, name=None)

.. note::
    该 API 的输入只能是带有 LoD 信息的 Tensor，如果要处理的输入是 Tensor 类型，请使用 :ref:`paddle.nn.functional.softmax <cn_api_paddle_nn_functional_softmax>`。

根据 LoD 信息将输入的第 0 维度进行划分，在划分的每一个区间内部进行运算。

对第 i 个区间内的元素的计算公式如下：

.. math::

    Out\left ( X[lod[i]:lod[i+1]],: \right ) = \frac{exp(X[lod[i]:lod[i+1],:])}{\sum (exp(X[lod[i]:lod[i+1],:]))}

输入 Tensor 的维度可为 :math:`[N，1]` 或者 :math:`[N]`，推荐使用 :math:`[N]` 。

例如，对有 6 个样本的 batch，每个样本的长度为 3，2，4，1，2，3，其 lod 信息为[[0, 3, 5, 9, 10, 12, 15]]，根据 lod 信息将第 0 维度划分为 6 份，在 :math:`X[0:3,:],X[3:5,:],X[5:9,:],X[9:10,:],X[10:12,:],X[12:15,:]`  中进行 softmax 运算。

::

     示例：

             给定：
                   input.data = [0.7, 1, 0.6,
                                 1.5, 1.1,
                                 1.2, 0.2, 0.6, 1.9,
                                 3.1,
                                 2.5, 0.8,
                                 0.1, 2.4, 1.3]
                   input.lod = [[0, 3, 5, 9, 10, 12, 15]]
              则：
                   output.data = [0.30724832, 0.41474187, 0.2780098,
                                  0.59868765, 0.40131235,
                                  0.2544242, 0.09359743, 0.13963096, 0.5123474,
                                  1.,
                                  0.84553474, 0.15446526,
                                  0.06995796, 0.69777346, 0.23226859]
                   output.lod = [[0, 3, 5, 9, 10, 12, 15]]


参数
:::::::::

    - **input** (Tensor) - 维度为 :math:`[N, 1]` 或者 :math:`[N]` 的 Tensor，推荐使用 :math:`[N]`。支持的数据类型：float32，float64。
    - **use_cudnn** (bool，可选) - 是否用 cudnn 核，仅当安装 cudnn 版本的 paddle 库且使用 gpu 训练或推理的时候生效。支持的数据类型：bool 型。默认值为 False。
    - **name**  (str，可选) – 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
根据区间计算 softmax 之后的 Tensor，其维度与 input 的维度一致，数据类型与 input 的数据类型一致。


代码示例
::::::::::::

COPY-FROM: paddle.static.nn.sequence_softmax
