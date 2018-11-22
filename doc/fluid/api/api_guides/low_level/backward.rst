########
反向传播
########


我们并不推荐直接调用 :code:`fluid` 中反向传播相关API。因为这是一个极底层的API。
请考虑使用 :ref:`api_guide_optimizer` 中的相关API替代反向传播 API 。

用户可以使用 :ref:`api_fluid_backward_append_backward` 中的 :code:`callback` 自
定义Operator的梯度计算形式。更多用法，请参考

* :ref:`api_fluid_backward_append_backward`