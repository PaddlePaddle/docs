.. _cn_api_paddle_nn_init_dirac_:

dirac\_
-------------------------------

.. py:class:: paddle.nn.init.dirac_(tensor, groups=1)

将输入张量的值设置为 Dirac delta 函数，该操作会直接修改输入张量。支持 3D、4D 和 5D 输入张量。

参数
::::::::::::

    - **tensor** (Tensor) - 输入张量。
    - **groups** (int，可选) - 输入张量的组数，默认值为 1。
