.. _api_guide_backward_en:


################
Back Propagation
################

The ability of neural network to define model depends on optimization algorithm. Optimization is a process of calculating gradient continuously and adjusting learnable parameters. You can refer to  :ref:`api_guide_optimizer_en` to learn more about optimization algorithm in Fluid.

In the training process of network, gradient calculation is divided into two steps: forward computing and `back propagation <https://en.wikipedia.org/wiki/Backpropagation>`_ .

Forward computing transfers the state of the input unit to the output unit according to the network structure you build.

Back propagation calculates the derivatives of two or more compound functions by means of `chain rule <https://en.wikipedia.org/wiki/Chain_rule>`_ . The gradient of output unit is propagated back to input unit. According to the calculated gradient, the learning parameters of the network are adjusted.


You could refer to `back propagation algorithm <http://deeplearning.stanford.edu/wiki/index.php/%E5%8F%8D%E5%90%91%E4%BC%A0%E5%AF%BC%E7%AE%97%E6%B3%95>`_ for detialed implementation process.

We do not recommend directly calling backpropagation-related APIs in  :code:`fluid` , as these are very low-level APIs. Consider using the relevant APIs in :ref:`api_guide_optimizer_en` instead. When you use optimizer APIs, Fluid automatically calculates the complex back-propagation for you.

If you want to implement it by yourself, you can also use: :code:`callback` in :ref:`api_fluid_backward_append_backward` to define the customized gradient form of Operator.
For more information, please refer to: :ref:`api_fluid_backward_append_backward`
