.. _cn_api_incubate_optimizer_functional_minimize_lbfgs:

minimize_lbfgs
-------------------------------

.. py:function:: paddle.incubate.optimizer.functional.minimize_lbfgs(objective_func, initial_position, history_size=100, max_iters=50, tolerance_grad=1e-08, tolerance_change=1e-08, initial_inverse_hessian_estimate=None, line_search_fn='strong_wolfe', max_line_search_iters=50, initial_step_length=1.0, dtype='float32', name=None)

使用 L-BFGS 方法求解可微函数 ``objective_func`` 的最小值。L-BFGS 是一种拟牛顿方法，用于解决可微函数上的无约束最优化问题。与之密切相关的是用于最优化的牛顿法,考虑迭代更新公式：

.. math::
    x_{k+1} = x_{k} + H_k \nabla{f_k}

如果 :math:`H_k` 是函数 :math:`f` 在 :math:`x_k`的逆海森矩阵, 此时就是牛顿法。如果 :math:`H_k` 满足对称性和正定性，用来作为逆海森矩阵的近似，则为高斯-牛顿法。在实际算法中，近似逆海森矩阵是通过整个或部分搜索历史的梯度计算得到，前者对应BFGS，后者对应于L-BFGS。


参考
:::::::::
    Jorge Nocedal, Stephen J. Wright, Numerical Optimization, Second Edition, 2006. pp179: Algorithm 7.5 (L-BFGS).

参数
:::::::::
    - **objective_func** (callable) - 待优化的目标函数，接受1维 Tensor 并返回一个标量。
    - **initial_position** (Tensor) - 迭代的初始位置，与 ``objective_func`` 的输入形状相同。 
    - **history_size** (Scalar，可选) - 指定储存的向量对{si,yi}数量。默认值：100。
    - **max_iters** (int，可选) - BFGS迭代的最大次数。默认值：50。
    - **tolerance_grad** (float，可选) - 当梯度的范数小于该值时，终止迭代。当前使用正无穷范数。默认值：1e-7。
    - **tolerance_change** (float，可选) - 当函数值/x值/其他参数 两次迭代的改变量小于该值时，终止迭代。默认值：1e-9。
    - **initial_inverse_hessian_estimate** (Tensor，可选) - 函数在初始位置时的近似逆海森矩阵，必须满足对称性和正定性。当为None时，将使用N阶单位矩阵，其中N为 ``initial_position`` 的size。默认值：None。
    - **line_search_fn** (str，可选) - 指定要使用的线搜索方法，目前只支持值为'strong wolfe'方法，未来将支持'Hager Zhang'方法。默认值：'strong wolfe'。
    - **max_line_search_iters** (int，可选) - 线搜索的最大迭代次数。默认值：50。
    - **initial_step_length** (float，可选) - 线搜索中第一次迭代时的步长，不同的初始步长可能会产生不同的优化结果。对于高斯牛顿类方法初始的试验步长应该总是1。默认值：1.0。
    - **dtype** ('float32' | 'float64'，可选) - 在算法中使用的数据类型，输入参数的数据类型必须与dtype保持一致。默认值：'float32'。
    - **name** (str，可选) - 操作名称。 更多信息请参考 :ref:`api_guide_Name`。默认值：None。

返回
:::::::::
    - is_converge (bool): 表示算法是否找到了满足容差的最小值。
    - num_func_calls (int): 目标函数被调用的次数。
    - position (Tensor): 最后一次迭代之后的位置，如果算法收敛，那么就是目标函数以初始位置开始迭代得到的最小值点。
    - objective_value (Tensor): 迭代终止位置的函数值。
    - objective_gradient (Tensor): 迭代终止位置的梯度。

代码示例
::::::::::
.. code-block:: python

    import paddle
            
    def func(x):
        return paddle.dot(x, x)

    x0 = paddle.to_tensor([1.3, 2.7])
    results = paddle.incubate.optimizer.functional.minimize_lbfgs(func, x0)
    print("is_converge: ", results[0])
    print("the minimum of func is: ", results[2])
    # is_converge:  is_converge:  Tensor(shape=[1], dtype=bool, place=Place(gpu:0), stop_gradient=True,
    #        [True])
    # the minimum of func is:  Tensor(shape=[2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
    #        [0., 0.])