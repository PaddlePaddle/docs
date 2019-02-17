.. _api_guide_learning_rate_scheduler_en:

########################
Learning rate scheduler
########################

When we use a method such as the gradient descent method to train the model, the training speed and loss are generally taken into consideration to select a relatively appropriate learning rate. However, if a fixed learning rate is used throughout the training process, the loss of the training set will not continue to decline after falling to a certain extent, but will 'jump' within a certain range. The jumping principle is shown in the figure below. When the loss function converges to the local minimum value, the update step will be too large due to the excessive learning rate. The parameter update will repeatedly *jump over* the local minimum value and an oscillation-like phenomenon will occur.

.. image:: ../../../images/learning_rate_scheduler.png
    :scale: 80 %
    :align: center


The learning rate scheduler defines a commonly used learning rate decay strategy to dynamically generate the learning rate. The learning rate decay function takes epoch or step as the parameter and returns a learning rate that gradually decreases with training. Thereby it reduces the training time and finds the local minimum value at the same time.

The following content describes the APIs related to the learning rate scheduler:

======

* :code:`noam_decay`: Noam decay. Please refer to `Attention Is All You Need <https://arxiv.org/pdf/1706.03762.pdf>`_ for related algorithms. For related API Reference please refer to :ref:`api_fluid_layers_noam_decay`

* :code:`exponential_decay`: Exponential decay. That is, each time the current learning rate is multiplied by the given decay rate to get the next learning rate. For related API Reference please refer to :ref:`api_fluid_layers_exponential_decay`

* :code:`natural_exp_decay`: Natural exponential decay. That is, each time the current learning rate is multiplied by the natural exponent of the given decay rate to get the next learning rate. For related API Reference please refer to :ref:`api_fluid_layers_natural_exp_decay`

* :code:`inverse_time_decay`: Inverse time decay. The decayed learning rate is inversely proportional to the current number of decays. For related API Reference please refer to :ref:`api_fluid_layers_inverse_time_decay`

* :code:`polynomial_decay`: Polynomial decay, i.e. the decayed learning rate is calculated in a polynomial format with the initial learning rate and the end learning rate. For related API Reference please refer to :ref:`api_fluid_layers_polynomial_decay`

* :code:`piecewise_decay`: Piecewise decay. That is, the stair-like decay for a given number of steps, the learning rate stays the same within each step. For related API Reference please refer to :ref:`api_fluid_layers_piecewise_decay`

* :code:`append_LARS`: The learning rate is obtained by the Layer-wise Adaptive Rate Scaling algorithm. For related algorithms, please refer to `Train Feedfoward Neural Network with Layerwise Adaptive Rate via Approximating Back-matching Propagation <https://arxiv. Org/abs/1802.09750>`_ . For related API Reference please refer to :ref:`api_fluid_layers_append_LARS`