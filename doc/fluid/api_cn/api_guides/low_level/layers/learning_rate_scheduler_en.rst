.. _api_guide_learning_rate_scheduler_en:

########################
Learning rate scheduler
########################

When we use a method such as the gradient descent method to train the model, the training speed and loss are generally taken into consideration to select a relatively appropriate learning rate. However, if a learning rate is used throughout the training process, the loss of the training set will not continue to decline after falling to a certain extent, but will oscillate within a certain range. The oscillation principle is shown in the figure below. When the loss function converges to the local minimum value, the update step will be too large due to the excessive learning rate. The parameter update will repeat the minimum value and the oscillation will occur repeatedly.

.. image:: ../../../../images/learning_rate_scheduler.png
    :scale: 80 %
    :align: center


The learning rate scheduler defines a commonly used learning rate decay strategy to dynamically generate the learning rate. The learning rate decay function takes epoch or step as a parameter and returns a learning rate that gradually decreases with training, thereby reducing the training time and local minimum. Values ​​can better optimize both aspects.

The following describes the related Api in the learning rate scheduler:

======

* :code:`noam_decay`: Nom decay, please refer to `Attention Is All You Need <https://arxiv.org/pdf/1706.03762.pdf>`_ for related algorithms.
  For related API Reference please refer to :ref:`api_fluid_layers_noam_decay`

* :code:`exponential_decay`: Exponential decay, that is, each time the current learning rate is multiplied by the given decay rate to get the next learning rate.
  For related API Reference please refer to :ref:`api_fluid_layers_exponential_decay`

* :code:`natural_exp_decay`: Natural exponential decay, that is, each time the current learning rate is multiplied by the natural exponent of the given decay rate to get the next learning rate.
  For related API Reference please refer to :ref:`api_fluid_layers_natural_exp_decay`

* :code:`inverse_time_decay`: Inverse time decay, ie the resulting learning rate is inversely proportional to the current number of decays.
  For related API Reference please refer to :ref:`api_fluid_layers_inverse_time_decay`

* :code:`polynomial_decay`: Polynomial decay, ie the resulting learning rate is the interpolation between the initial learning rate and the given final learning by the polynomial calculation weights.
  For related API Reference please refer to :ref:`api_fluid_layers_polynomial_decay`

* :code:`piecewise_decay`: Segmentation decay, that is, the stepwise decay of a given number of segments, the learning rate is the same within each segment.
  For related API Reference please refer to :ref:`api_fluid_layers_piecewise_decay`

* :code:`append_LARS`: The learning rate is obtained by the Layer-wise Adaptive Rate Scaling algorithm. For related algorithms, please refer to "Train Feedfoward Neural Network with Layer-wise Adaptive Rate via Approximating Back-matching Propagation" <https://arxiv. Org/abs/1802.09750>`_.
  For related API Reference please refer to :ref:`api_fluid_layers_append_LARS`