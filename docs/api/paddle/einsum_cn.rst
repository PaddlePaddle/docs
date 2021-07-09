.. _cn_api_tensor_einsum:

einsum
------

.. py:function:: paddle:einsum(equation, *operands)

该函数用于对一组输入张量进行 Einstein 求和。

Einstein 求和是一种采用 Einstein 标记法描述的张量求和方式，输入单个或多个张量，输出单个张量。

Einstein 求和覆盖了如下特殊形式的张量计算：
    1. 单操作数：
        - 迹：trace
        - 对角元：diagonal
        - 转置：transpose
        - 求和：sum
    2. 双操作数
        - 内积：dot
        - 外积：outer
        - 广播乘积：mul，*
        - 矩阵乘：matmul
        - 批量矩阵乘：bmm
    3. 多操作数
        - 广播乘积：mul，*
        - 多矩阵乘：A.matmul(B).matmul(C)

**关于求和标记的约定**

    - 张量的坐标轴下标使用英文字母表示，不区分大小写，如'ijk'表示张量的三个坐标轴分别为i，j和k
    - 用于广播的坐标轴用省略号`...`表示，如'i...j'表示将对张量中除首末坐标轴的维度进行广播，广播维度默认位于输出张量的高维位置
    - 标记中仅出现一个的下标为自由标，重复出现的下标为哑标，输入张量需要以自由标为索引对齐，将标量维度规约为标量
    - 输入张量的标记按顺序用`,`分开
    - 用户允许使用输出标记定制化输出，有效的输出标记需满足如下约定规则：
        - 输出标记位于`->`右侧
        - 输出标记为空时，返回输出张量的全量求和结果
        - 若输出包含广播维度，则输出标记需包含`...`
        - 输出可包含自由标和哑标，但不能包含输入标记中未出现的下标
        - 输出下标不可以重复出现

**关于trace和diagonal的标记约定(待实现功能)**

    - 在单个输入张量的标记中重复出现的下标称为对角标，对角标对应的坐标轴需进行对角化操作，如'i...i'表示需对首尾坐标轴进行对角化
    - 若无输出标记或输出标记中不包含对角标，则对角标对应维度规约为标量，相应维度取消，等价于trace操作
    - 若输出标记中包含对角标，则保留对角标维度，等价于diagonal操作

参数
：：
    **equation** (str): 求和标记
    
    **operands** (Tensor, [Tensor, ...]): 输入张量

返回
：：
    Tensor: 输出张量

代码示例
:::::::::

    .. code-block::
        import paddle
        import numpy as np
        np.random.seed(102)
        x = paddle.to_tensor(np.random.rand(4))
        y = paddle.to_tensor(np.random.rand(5))
        # sum
        print(paddle.einsum('i->', x))
        # Tensor(shape=[], dtype=float64, place=CUDAPlace(0), stop_gradient=True, 2.30369050)
        # dot
        print(paddle.einsum('i,i->', x, x))
        # Tensor(shape=[], dtype=float64, place=CUDAPlace(0), stop_gradient=True, 1.43773247)
        # outer
        print(paddle.einsum("i,j->ij", x, y)),
        # Tensor(shape=[4, 5], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
        #         [[0.34590188, 0.48353496, 0.09996135, 0.18656330, 0.21392910],
        #         [0.39122025, 0.54688535, 0.11305780, 0.21100591, 0.24195704],
        #         [0.17320613, 0.24212422, 0.05005442, 0.09341929, 0.10712238],
        #         [0.42290818, 0.59118179, 0.12221522, 0.22809690, 0.26155500]])
        A = paddle.to_tensor(np.random.rand(2, 3, 2))
        B = paddle.to_tensor(np.random.rand(2, 2, 3))
        # transpose
        print(paddle.einsum('ijk->kji', A))
        #  Tensor(shape=[2, 3, 2], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
        #        [[[0.49174730, 0.33344683],
        #          [0.89440989, 0.26162022],
        #          [0.36116209, 0.12241719]],
        #         [[0.49019824, 0.51895050],
        #          [0.18241053, 0.13092809],
        #          [0.81059146, 0.55165734]]])
        # batch matrix multiplication
        print(paddle.einsum('ijk, ikl->ijl', A,B))
        # Tensor(shape=[2, 3, 3], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
        #     [[[0.13654339, 0.39331432, 0.65059661],
        #      [0.07171420, 0.57518653, 0.77629221],
        #      [0.21250688, 0.37793541, 0.73643411]],
        #     [[0.56925339, 0.65859030, 0.57509818],
        #      [0.30368265, 0.25778348, 0.21630400],
        #      [0.39587265, 0.58031243, 0.51824755]]])
        # Ellipsis transpose
        print(paddle.einsum('...jk->...kj', A))
        # Tensor(shape=[2, 2, 3], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
        #     [[[0.49174730, 0.89440989, 0.36116209],
        #         [0.49019824, 0.18241053, 0.81059146]],
        #         [[0.33344683, 0.26162022, 0.12241719],
        #         [0.51895050, 0.13092809, 0.55165734]]])
        # Ellipsis batch matrix multiplication
        print(paddle.einsum('...jk, ...kl->...jl', A,B))
        # Tensor(shape=[2, 3, 3], dtype=float64, place=CUDAPlace(0), stop_gradient=True,
        # [[[0.13654339, 0.39331432, 0.65059661],
        #     [0.07171420, 0.57518653, 0.77629221],
        #     [0.21250688, 0.37793541, 0.73643411]],
        #     [[0.56925339, 0.65859030, 0.57509818],
        #     [0.30368265, 0.25778348, 0.21630400],
        #     [0.39587265, 0.58031243, 0.51824755]]])