.. _api_guide_loss_function_en:

##############
Loss function
##############

The loss function defines the difference between the inference result and the ground-truth result. As the optimization target, it directly determines whether the model training is good or not, and many researches also focus on the optimization of the loss function design.
Paddle Fluid offers diverse types of loss functions for a variety of tasks. Let's take a look at the commonly-used loss functions included in Paddle Fluid.

Regression
===========

The squared error loss uses the square of the error between the predicted value and the ground-truth value as the sample loss, which is the most basic loss function in the regression problems.
For API Reference,  please refer to :ref:`api_fluid_layers_square_error_cost`.

Smooth L1 loss (smooth_l1 loss) is a piecewise loss function that is relatively insensitive to outliers and therefore more robust.
For API Reference,  please refer to :ref:`api_fluid_layers_smooth_l1`.


Classification
================

`cross entropy <https://en.wikipedia.org/wiki/Cross_entropy>`_ is the most widely used loss function in classification problems.  The interfaces in Paddle Fluid for the cross entropy loss functions are divided into the one accepting fractional input of normalized probability values ​​and another for non-normalized input. And Fluid supports two types labels, namely soft label and hard label.
For API Reference,  please refer to :ref:`api_fluid_layers_cross_entropy` and :ref:`api_fluid_layers_softmax_with_cross_entropy`.

Multi-label classification
----------------------------
For the multi-label classification, such as the occasion that an article belongs to multiple categories like politics, technology, it is necessary to calculate the loss by treating each category as an independent binary-classification problem. We provide the sigmoid_cross_entropy_with_logits loss function for this purpose.
For API Reference,  please refer to :ref:`api_fluid_layers_sigmoid_cross_entropy_with_logits`.

Large-scale classification
-----------------------------
For large-scale classification problems, special methods and corresponding loss functions are usually needed to speed up the training. The commonly used methods are
`Noise contrastive estimation (NCE) <http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf>`_ and `Hierarchical sigmoid <http://www.iro.umontreal.ca/~lisa/pointeurs/hierarchical-nnlm-aistats05.pdf>`_ .

* NCE solves the binary-classification problem of discriminating the true distribution and the noise distribution by converting the multi-classification problem into a classifier. The maximum likelihood estimation is performed based on the binary-classification to avoid calculating the normalization factor in the full-class space to reduce computational complexity.
* Hierarchical sigmoid realizes multi-classification by hierarchical classification of binary trees. The loss of each sample corresponds to the sum of the cross-entropy of the binary-classification for each node on the coding path, which avoids the calculation of the normalization factor and reduces the computational complexity.
The loss functions for both methods are available in Paddle Fluid. For API Reference please refer to :ref:`api_fluid_layers_nce` and :ref:`api_fluid_layers_hsigmoid`.

Sequence classification
-------------------------
Sequence classification can be divided into the following three types:

* Sequence Classification problem is that the entire sequence corresponds to a prediction label, such as text classification. This is a common classification problem, you can use cross entropy as the loss function.
* Segment Classification problem is that each segment in the sequence corresponds to its own category tag, such as named entity recognition. For this sequence labeling problem, `the (Linear Chain) Conditional Random Field (CRF) <http://www.cs.columbia.edu/~mcollins/fb.pdf>`_ is a commonly used model. The method uses the likelihood probability on the sentence level, and the labels for different positions in the sequence are no longer conditionally independent, which can effectively solve the label offset problem. Support for CRF loss functions is available in Paddle Fluid. For API Reference please refer to :ref:`api_fluid_layers_linear_chain_crf` .
* Temporal Classification problem needs to label unsegmented sequences, such as speech recognition. For this time-based classification problem, `CTC(Connectionist Temporal Classification) <http://people.idsia.ch/~santiago/papers/icml2006.pdf>`_ loss function does not need to align input data and labels, and is able to perform end-to-end training. Paddle Fluid provides a warpctc interface to calculate the corresponding loss. For API Reference,  please refer to :ref:`api_fluid_layers_warpctc` .

Rank
=========

`Rank problems <https://en.wikipedia.org/wiki/Learning_to_rank>`_ can use learning methods of Pointwise, Pairwise, and Listwise. Different methods require different loss functions:

* The Pointwise method solves the ranking problem by approximating the regression problem. Therefore the loss function of the regression problem can be used.
* Pairwise's method requires a special loss function. Pairwise solves the sorting problem by approximating the classification problem, using relevance score of two documents and the query to use the partial order as the binary-classification label to calculate the loss. Paddle Fluid provides two commonly used loss functions for Pairwise methods. For API Reference please refer to :ref:`api_fluid_layers_rank_loss` and :ref:`api_fluid_layers_margin_rank_loss`.

More
====

For more complex loss functions, try to use combinations of other loss functions; the :ref:`api_fluid_layers_dice_loss` provided in Paddle Fluid for image segmentation tasks is an example of using combinations of other operators  (calculate the average likelihood probability of each pixel position). The multi-objective loss function can also be considered similarly, such as Faster RCNN that uses the weighted sum of cross entropy and smooth_l1 loss as a loss function.

**Note**, after defining the loss function, in order to optimize with :ref:`api_guide_optimizer_en`, you usually need to use :ref:`api_fluid_layers_mean` or other operations to convert the high-dimensional Tensor returned by the loss function to a Scalar value.
