.. _api_guide_backward:


########
反向传播
########

神经网络对模型的表达能力依赖于优化算法，优化是一个不断计算梯度并调整可学习参数的过程，Fluid 中的优化算法可参考 :ref:`api_guide_optimizer` 。

在网络的训练过程中，梯度计算分为两个步骤：前向计算与 `反向传播 <https://en.wikipedia.org/wiki/Backpropagation>`_ 。

- 前向计算会根据您搭建的网络结构，将输入单元的状态传递到输出单元。

- 反向传播借助 `链式法则 <https://en.wikipedia.org/wiki/Chain_rule>`_ ，计算两个或两个以上复合函数的导数，将输出单元的梯度反向传播回输入单元，根据计算出的梯度，调整网络的可学习参数。

详细实现过程可以参考阅读 `反向传导算法 <http://deeplearning.stanford.edu/wiki/index.php/%E5%8F%8D%E5%90%91%E4%BC%A0%E5%AF%BC%E7%AE%97%E6%B3%95>`_ 。

在 Fluid 中，我们并不推荐直接调用 :code:`fluid` 中反向传播相关 API，因为这是一个极底层的 API，请考虑使用 :ref:`api_guide_optimizer` 中的相关 API 替代。当您使用优化相关 API 时，Fluid 会自动为您计算复杂的反向传播过程。

如想自己实现，您也可以使用 :ref:`cn_api_fluid_backward_append_backward` 中的 :code:`callback` 自
定义 Operator 的梯度计算形式。更多用法，请参考：

* :ref:`cn_api_fluid_backward_append_backward`
