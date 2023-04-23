.. _cn_api_incubate_optimizer_functional_minimize_lbfgs:

minimize_lbfgs
-------------------------------

.. py:function:: paddle.incubate.optimizer.functional.minimize_lbfgs(objective_func, initial_position, history_size=100, max_iters=50, tolerance_grad=1e-08, tolerance_change=1e-08, initial_inverse_hessian_estimate=None, line_search_fn='strong_wolfe', max_line_search_iters=50, initial_step_length=1.0, dtype='float32', name=None)

``minimize_lbfgs`` 使用 L-BFGS 方法求解可微函数 ``objective_func`` 的最小值。

L-BFGS 是限制内存的 BFGS 方法，适用于海森矩阵为稠密矩阵、内存开销较大场景。BFGS 参考 :ref:`cn_api_incubate_optimizer_functional_minimize_bfgs` .

LBFGS 具体原理参考书籍 Jorge Nocedal, Stephen J. Wright, Numerical Optimization, Second Edition, 2006. pp179: Algorithm 7.5 (L-BFGS).


使用方法
:::::::::
- minimize_bfgs 优化器当前实现为函数形式，与 Paddle 现有 SGD、Adam 优化器等使用略微有些区别。
  SGD/Adam 等通过调用 backward()计算梯度，并使用 step()更新网络参数，而 minimize_lbfgs 传入
  loss 函数，并返回优化后参数，返回参数需要通过 :ref:`cn_api_paddle_tensor_creation_assign` 以 inpalce 方式进行更新。具体参考代码示例 1.
- 由于当前实现上一些限制，当前 minimize_bfgs 要求函数输入为一维 Tensor。当输入参数维度超过一维，
  可以先将参数展平，使用 minimize_bfgs 计算后，再 reshape 到原有形状，更新参数。具体参考代码示例 2.


.. warning::
  该 API 目前为 Beta 版本，函数签名在未来版本可能发生变化。

.. note::
  当前仅支持动态图模式下使用。

.. note::
  当前仅支持 Vector-scalar 形式函数，即目标函数输入为一维 Tensor，输出为只包含一个元素 Tensor。

参数
:::::::::
    - **objective_func** (callable) - 待优化的目标函数，接受 1 维 Tensor 并返回一个标量。
    - **initial_position** (Tensor) - 迭代的初始位置，与 ``objective_func`` 的输入形状相同。
    - **history_size** (Scalar，可选) - 指定储存的向量对{si,yi}数量。默认值：100。
    - **max_iters** (int，可选) - BFGS 迭代的最大次数。默认值：50。
    - **tolerance_grad** (float，可选) - 当梯度的范数小于该值时，终止迭代。当前使用正无穷范数。默认值：1e-7。
    - **tolerance_change** (float，可选) - 当函数值/x 值/其他参数 两次迭代的改变量小于该值时，终止迭代。默认值：1e-9。
    - **initial_inverse_hessian_estimate** (Tensor，可选) - 函数在初始位置时的近似逆海森矩阵，必须满足对称性和正定性。当为 None 时，将使用 N 阶单位矩阵，其中 N 为 ``initial_position`` 的 size。默认值：None。
    - **line_search_fn** (str，可选) - 指定要使用的线搜索方法，目前只支持值为'strong wolfe'方法，未来将支持'Hager Zhang'方法。默认值：'strong wolfe'。
    - **max_line_search_iters** (int，可选) - 线搜索的最大迭代次数。默认值：50。
    - **initial_step_length** (float，可选) - 线搜索中第一次迭代时的步长，不同的初始步长可能会产生不同的优化结果。对于高斯牛顿类方法初始的试验步长应该总是 1。默认值：1.0。
    - **dtype** ('float32' | 'float64'，可选) - 在算法中使用的数据类型，输入参数的数据类型必须与 dtype 保持一致。默认值：'float32'。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
    - is_converge (bool)：表示算法是否找到了满足容差的最小值。
    - num_func_calls (int)：目标函数被调用的次数。
    - position (Tensor)：最后一次迭代之后的位置，如果算法收敛，那么就是目标函数以初始位置开始迭代得到的最小值点。
    - objective_value (Tensor)：迭代终止位置的函数值。
    - objective_gradient (Tensor)：迭代终止位置的梯度。



代码示例 1：
::::::::::

.. code-block:: python

    import paddle


    # 随机模拟一批输入数据
    inputs = paddle.normal(shape=(100, 1))
    labels = inputs * 2.0

    # 定义 loss 函数
    def loss(w):
        y = w * inputs
        return paddle.nn.functional.square_error_cost(y, labels).mean()

    # 初始化权重参数
    w = paddle.normal(shape=(1,))

    # 调用 bfgs 方法求解使得 loss 最小的权重，并更新参数
    for epoch in range(0, 10):
        # 调用 bfgs 方法优化 loss，注意返回的第三个参数表示权重
        w_update= paddle.incubate.optimizer.functional.minimize_lbfgs(loss, w)[2]
        # 使用 paddle.assign，以 inplace 方式更新参数
        paddle.assign(w_update, w)


代码示例 2：输入参数维度超过一维
::::::::::

.. code-block:: python

    import paddle


    def flatten(x):
        return x.flatten()


    def unflatten(x):
        return x.reshape((2,2))


    # 假设网络参数超过一维
    def net(x):
        assert len(x.shape) > 1
        return x.square().mean()


    # 待优化函数
    def bfgs_f(flatten_x):
        return net(unflatten(flatten_x))


    x = paddle.rand([2,2])
    for i in range(0, 10):
        # 使用 minimize_lbfgs 前，先将 x 展平
        x_update = paddle.incubate.optimizer.functional.minimize_lbfgs(bfgs_f, flatten(x))[2]
        # 将 x_update unflatten，然后更新参数
        paddle.assign(unflatten(x_update), x)
