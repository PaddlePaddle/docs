.. _cn_api_paddle_argwhere:

argwhere
-------------------------------

.. py:function:: paddle.argwhere(input)




返回输入  ``x``  中非零元素的坐标。如果输入  ``x``  有  ``n``  维，共包含  ``z``  个非零元素，返回结果是一个  ``shape``  等于  ``[z x n]``  的  ``Tensor`` ，第  ``i``  行代表输入中第  ``i``  个非零元素的坐标。

参数
:::::::::

    - **input** （Tensor）– 输入的 Tensor。



返回
:::::::::
    - **Tensor(1-D Tensor)**，数据类型为 **INT64** 。



代码示例
:::::::::

COPY-FROM: paddle.argwhere
