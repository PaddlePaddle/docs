.. _cn_api_paddle_vision_models_shufflenetv2_x1_5:

shufflenetv2_x1_5
-------------------------------

.. py:function:: paddle.vision.models.shufflenetv2_x1_5(pretrained=False, **kwargs)

 shufflenetv2_x1_5模型，来自论文 `"ShuffleNet V2: Practical Guidelines for Ecient CNN Architecture Design" <https://arxiv.org/pdf/1807.11164.pdf>`_。

参数
:::::::::
  - **pretrained** (bool，可选) - 是否加载在imagenet数据集上的预训练权重。默认值：False。

返回
:::::::::
shufflenetv2_x1_5模型，Layer的实例。

代码示例
:::::::::

.. code-block:: python

    import paddle
    from paddle.vision.models import shufflenetv2_x1_5
    # build model
    model = shufflenetv2_x1_5()
    # build model and load imagenet pretrained weight
    # model = shufflenetv2_x1_5(pretrained=True)
    x = paddle.rand([1, 3, 224, 224])
    out = model(x)
    print(out.shape)
