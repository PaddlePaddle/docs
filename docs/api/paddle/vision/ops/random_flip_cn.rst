.. _cn_api_paddle_vision_ops_random_flip:

random_flip
-------------------------------

.. py:function:: paddle.vision.ops.random_flip(x, prob=0.5, name=None)

为输入张量生成是否需要翻转图像的指示Tensor，将输入Tensor的第1维度作为批次大小，并对每个样本生成一个布尔值表示是否翻转该样本。

参数
:::::::::
    - x (Tensor) - 形状为[N, ...]的Tensor，N为批次大小，用于生成形状为[N, 1]的输出Tensor。
    - prob (float) - 翻转输入样本的概率，取值在0到1之间，默认为0.5
    - name (str，可选）- 默认值为None。一般用户无需设置，具体用法请参见 :ref:`api_guide_Name`。

返回
:::::::::
    形状为[N, 1]的布尔类型Tensor，N为输入Tensor的第1维的形状

代码示例
:::::::::

..  code-block:: python

    import paddle

    x = paddle.rand(shape=[8, 3, 32, 32])
    mirror = paddle.vision.ops.random_flip(x)

    print(mirror)
