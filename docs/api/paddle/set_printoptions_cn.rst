.. _cn_api_tensor_set_printoptions:

set_printoptions
-------------------------------

.. py:function:: paddle.set_printoptions(precision=None, threshold=None, edgeitems=None, sci_mode=None, linewidth=None)



设置 paddle 中 ``Tensor`` 的打印配置选项。

参数
:::::::::
    - **precision** (int，可选) - 浮点数的小数位数，默认值为 8。
    - **threshold** (int，可选) - 打印的元素个数上限，默认值为 1000。
    - **edgeitems** (int，可选) - 以缩略形式打印时左右两边的元素个数，默认值为 3。
    - **sci_mode** (bool，可选) - 是否以科学计数法打印，默认值为 False。
    - **linewidth** (int，可选) – 每行的字符数，默认值为 80。


返回
:::::::::
无。


代码示例
:::::::::

COPY-FROM: paddle.set_printoptions
