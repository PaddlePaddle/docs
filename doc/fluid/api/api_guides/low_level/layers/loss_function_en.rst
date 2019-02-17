.. _api_guide_loss_function_en:

##############
Loss function
##########3###

The loss function defines the difference between the fitting result and the real result. As the optimization target, it directly determines whether the model training is good or not, and the content of many research work is also focused on the design optimization of the loss function.
Paddle Fluid offers several types of loss functions for a variety of tasks. The following are some of the more commonly used loss functions included in Paddle Fluid.

Regression
===========

The squared error loss uses the square of the error between the predicted value and the ground-truth value as the sample loss, which is the most basic loss function in the regression problem.
About API Reference,  please refer to :ref:`api_fluid_layers_square_error_cost`.

Smooth L1 loss (smooth_l1 loss) is a piecewise loss function that is relatively insensitive to outliers and therefore more robust.
About API Reference,  please refer to :ref:`api_fluid_layers_smooth_l1`.


Classification
================

`cross entropy <https://en.wikipedia.org/wiki/Cross_entropy>`_ is the most widely used loss function in classification problems, and accepts normalized probability values ​​and non-normalized in Paddle Fluid. The interface for the two cross entropy loss functions of the fractional input, and supports two label category labels, soft label and hard label.
About API Reference,  please refer to :ref:`api_fluid_layers_cross_entropy` and :ref:`api_fluid_layers_softmax_with_cross_entropy`.

Multi-label classification
----------------------------
For the multi-label classification problem, if an article belongs to multiple categories of politics, technology, etc., it is necessary to calculate the loss as an independent two-class problem. Paddle Fluid provides the sigmoid_cross_entropy_with_logits loss function for this purpose.
About API Reference,  please refer to :ref:`api_fluid_layers_sigmoid_cross_entropy_with_logits`.

Large-scale classification
-----------------------------
For large-scale classification problems, special methods and corresponding loss functions are usually needed to speed up the training. The commonly used methods are Noise-contrasive estimation (NCE) <http://proceedings.mlr.press/v9/gutmann10a /gutmann10a.pdf>`_ and `level sigmoid <http://www.iro.umontreal.ca/~lisa/pointeurs/hierarchical-nnlm-aistats05.pdf>`_.

* Noise contrast estimation By discriminating the multi-classification problem into a learning classifier to discriminate the two-class problem of data from the true distribution and the noise distribution, the maximum likelihood estimation is based on the bipatition tasks to avoid calculating the normalization factor in the full-class space. Reduced computational complexity.
* Hierarchical sigmoid Multi-classification is realized by hierarchical classification of binary trees. The loss of each sample corresponds to the sum of the cross-entropy of the two-class classification on the coding path, which avoids the calculation of the normalization factor and reduces the computational complexity.
The loss functions for both methods are available in Paddle Fluid. For API Reference please refer to :ref:`api_fluid_layers_nce` and :ref:`api_fluid_layers_hsigmoid`.

Sequence classification
-------------------------
Sequence classification can be divided into the following three types:

* Sequence Classification problem, the entire sequence corresponds to a prediction label, such as text classification. This is a common classification problem, you can use cross entropy as a loss function.
* Segment Classification problem, each segment in the sequence corresponds to its own category tag, such as named entity recognition. For this sequence labeling problem, the (Linear Chain) Conditional Random Field (CRF) <http://www.cs.columbia.edu/~mcollins/fb.pdf>`_ is a commonly used model. The method uses the likelihood probability of the sentence level, and the labels in different positions in the sequence are no longer conditionally independent, which can effectively solve the label offset problem. Support for CRF-corresponding loss functions is available in Paddle Fluid. For API Reference please refer to :ref:`api_fluid_layers_linear_chain_crf`.
* Temporal Classification problem, which needs to label undivided sequences, such as speech recognition. For this timing classification problem, `CTC(Connectionist Temporal Classification) <http://people.idsia.ch/~santiago/papers/icml2006.pdf>`_ loss function does not need to align input data and labels, End training, Paddle Fluid provides a warpctc interface to calculate the corresponding loss. About API Reference,  please refer to: ref: `api_fluid_layers_warpctc`.

Rank
====

`Rank problems <https://en.wikipedia.org/wiki/Learning_to_rank>`_ You can use Pointwise, Pairwise, and Listwise learning methods. Different methods require different loss functions:

* The Pointwise method solves the sorting problem by approximating the regression problem, and the loss function of the regression problem can be used.
* Pairwise's method requires a specially designed loss function that solves the sorting problem by approximating the classification problem, using two documents and the query's relevance score to use the partial order as the two-category label to calculate the loss. Paddle Fluid provides loss functions for two commonly used Pairwise methods. For API Reference please refer to :ref:`api_fluid_layers_rank_loss` and :ref:`api_fluid_layers_margin_rank_loss`.

More
====

For some more complex loss functions, try using other loss function combinations; the :ref:`api_fluid_layers_dice_loss` provided in Paddle Fluid for image segmentation tasks is to use other OP combinations (calculate the mean of each pixel position likelihood probability). The multi-objective loss function can also be considered as a case where the Faster RCNN uses the weighted sum of cross entropy and smooth_l1 loss as a loss function.

**Note**, after defining the loss function, to be able to optimize with :ref:`api_guide_optimizer`, you usually need to use :ref:`api_fluid_layers_mean` or other operations to convert the high-dimensional Tensor returned by the loss function to a Scalar value.
