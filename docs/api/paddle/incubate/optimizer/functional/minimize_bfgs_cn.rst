.. _cn_api_incubate_optimizer_functional_minimize_bfgs:

minimize_bfgs
-------------------------------

.. py:function:: paddle.incubate.optimizer.functional.minimize_bfgs(objective_func, initial_position, max_iters=50, tolerance_grad=1e-07, tolerance_change=1e-09, initial_inverse_hessian_estimate=None, line_search_fn='strong_wolfe', max_line_search_iters=50, initial_step_length=1.0, dtype='float32', name=None)


使用 BFGS 方法求解可微函数 ``objective_func`` 的最小值。BFGS 是一种拟牛顿方法，用于解决可微函数上的无约束最优化问题。与之密切相关的是用于最优化的牛顿法,考虑迭代更新公式：

.. math::
    x_{k+1} = x_{k} + H_k \nabla{f_k}


如果 :math:`H_k` 是函数 :math:`f` 在 :math:`x_k`的逆海森矩阵, 此时就是牛顿法。如果 :math:`H_k` 满足对称性和正定性，用来作为逆海森矩阵的近似，则为高斯-牛顿法。在实际算法中，近似逆海森矩阵是通过整个或部分搜索历史的梯度计算得到，前者对应BFGS，后者对应于L-BFGS。


参考
:::::::::
    Jorge Nocedal, Stephen J. Wright, Numerical Optimization, Second Edition, 2006. pp140: Algorithm 6.1 (BFGS Method).

参数
:::::::::
    - objective_func: 待优化的目标函数. 接受多元输入并返回一个标量。
    - initial_position (Tensor): 迭代的初始位置。 
    - max_iters (int): BFGS迭代的最大次数。
    - tolerance_grad (float): 当梯度的范数小于该值时，终止迭代。当前使用正无穷范数。
    - tolerance_change (float): 当函数值/x值/其他参数 两次迭代的改变量小于该值时，终止迭代。
    - initial_inverse_hessian_estimate (Tensor): 函数在初始位置时的近似逆海森矩阵，必须满足对称性和正定性。
    - line_search_fn (str): 指定要使用的线搜索方法，目前只支持值为'strong wolfe'方法，未来将支持'Hager Zhang'方法。
    - max_line_search_iters (int): 线搜索的最大迭代次数。
    - initial_step_length (float): 线搜索中第一次迭代时的步长，不同的初始步长可能会产生不同的优化结果。对于高斯牛顿类方法初始的试验步长应该总是1。
    - dtype ('float32' | 'float64'): 在算法中使用的数据类型。

返回
:::::::::
    - is_converge (bool): 表示算法是否找到了满足容差的最小值。
    - num_func_calls (int): 目标函数被调用的次数。
    - position (Tensor): 最后一次迭代之后的位置，如果算法收敛，那么就是目标函数以初始位置开始迭代得到的最小值点。
    - objective_value (Tensor): 迭代终止位置的函数值。
    - objective_gradient (Tensor): 迭代终止位置的梯度。
    - inverse_hessian_estimate (Tensor): 迭代终止位置的近似逆海森矩阵。

代码示例
::::::::::
.. code-block:: python

    import paddle

    def func(x):
        return paddle.dot(x, x)

    x0 = paddle.to_tensor([1.3, 2.7])
    results = paddle.incubate.optimizer.functional.minimize_bfgs(func, x0)
    print("is_converge: ", results[0])
    print("the minimum of func is: ", results[2])
    # is_converge:  is_converge:  Tensor(shape=[1], dtype=bool, place=Place(gpu:0), stop_gradient=True,
    #        [True])
    # the minimum of func is:  Tensor(shape=[2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
    #        [0., 0.])