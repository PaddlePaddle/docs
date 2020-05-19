# Release Note

## Important Statements

- This version is a beta version. It is still in iteration and is not stable at present. Incompatible upgrade may be subsequently performed on APIs based on the feedback. For developers who want to experience the latest features of Paddle, welcome to this version. For industrial application scenarios requiring high stability, the stable Paddle Version 1.8 is recommended.

- This version mainly popularizes the dynamic graph development method and provides the encapsulation of high-level APIs. The dynamic graph mode has great flexibility and high-level APIs can greatly reduces duplicated codes. For beginners or basic task scenarios, the high-level API development method is recommended because it is simple and easy to use. For senior developers who want to implement complex functions, the dynamic graph API is commended because it is flexible and efficient.

- This version also optimizes the Paddle API directory system. The APIs in the original directory can create an alias and are still available, but it is recommended that new programs use the new directory structure.

## Basic Framework

### Basic APIs

- Networking APIs achieve dynamic and static unity and support operation in dynamic and static graph modes

- The API directory structure is adjusted. In the Paddle Version 1.x, the APIs are mainly located in the paddle.fluid directory. This version adjusts the API directory structure so that the classification is more reasonable. The specific adjustment rules are as follows:

  - Moves the APIs related to the tensor operations in the original fluid.layers directory to the paddle.tensor directory
  - Moves the networking-related operations in the original fluid.layers directory to the paddle.nn directory. Puts the types with parameters in the paddle.nn.layers directory and the functional APIs in the paddle.nn.functional directory
  - Moves the special API for dynamic graphs in the original fluid.dygraph directory to the paddle.imperative directory
  - Creates a paddle.framework directory that is used to store framework-related program, executor, and other APIs
  - Creates a paddle.distributed directory that is used to store distributed related APIs
  - Creates a paddle.optimizer directory that is used to store APIs related to optimization algorithms
  - Creates a paddle.metric directory that is used to create APIs related to evaluation index calculation
  - Creates a paddle.incubate directory that is used to store incubating codes. APIs may be adjusted. This directory stores codes related to complex number computation and high-level APIs
  - Creates an alias in the paddle directory for all APIs in the paddle.tensor and paddle.framework directories. For example, paddle.tensor.creation.ones can use paddle.ones as an alias

- The added APIs are as follows:

  - Adds eight networking APIs in the paddle.nn directory: interpolate, LogSoftmax, ReLU, Sigmoid, loss.BCELoss, loss.L1Loss, loss.MSELoss, and loss.NLLLoss
  - Adds 59 tensor-related APIs in the paddle.tensor directory: add, addcmul, addmm, allclose, arange, argmax, atan, bmm, cholesky, clamp, cross, diag\_embed, dist, div, dot, elementwise\_equal, elementwise\_sum, equal, eye, flip, full, full\_like, gather, index\_sample, index\_select, linspace, log1p, logsumexp, matmul, max, meshgrid, min, mm, mul, nonzero, norm, ones, ones\_like, pow, randint, randn, randperm, roll, sin, sort, split, sqrt, squeeze, stack, std, sum, t, tanh, tril, triu, unsqueeze, where, zeros, and zeros\_like
  - Adds device\_guard that is used to specify a device. Adds manual\_seed that is used to initialize a random number seed

### High-level APIs

- Adds a paddle.incubate.hapi directory. Encapsulates common operations such as networking, training, evaluation, inference, and access during the model development process. Implements low-code development. Uses the dynamic graph implementation mode of MNIST task comparison. High-level APIs can reduce 80% of executable codes.
- Adds model-type encapsulation. Inherits the layer type. Encapsulates common basic functions during the model development process, including:
  - Provides a prepare API that is used to specify a loss function and an optimization algorithm
  - Provides a fit API to implement training and evaluation. Implements the execution of model storage and other user-defined functions during the training process by means of callback
  - Provides an evaluate interface to implement the inference and evaluation index calculation on the evaluation set
  - Provides a predict interface to implement specific test data inference
  - Provides a train\_batch interface to implement the training of single-batch data
- Adds a dataset interface to encapsulate commonly-used data sets and supports random access to data
- Adds encapsulation of common Loss and Metric types
- Adds 16 common data processing interfaces including Resize and Normalize in the CV field
- Adds lenet, vgg, resnet, mobilenetv1, and mobilenetv2 image classification backbone networks in the CV field
- Adds MultiHeadAttention, BeamSearchDecoder, TransformerEncoder, TransformerDecoder, and DynamicDecode APIs in the NLP field
- Releases 12 models based on high-level API implementation, including Transformer, Seq2seq, LAC, BMN, ResNet, YOLOv3, VGG, MobileNet, TSM, CycleGAN, Bert, and OCR

### Performance Optimization

- Adds a `reshape+transpose+matmul` fuse so that the performance of the INT8 model is improved by about 4% (on the 6271 machine) after Ernie quantization. After the quantization, the speed of the INT8 model is increased by about 6.58 times compared with the FP32 model on which DNNL optimization (including fuses) and quantization are not performed

### Debugging Analysis

- To solve the problem of program printing contents being too lengthy and low utilization efficiency during debugging, considerably simplifies the printing strings of objects such as programs, blocks, operators, and variables, thus improving the debugging efficiency without losing effective information
- To solve the problem of insecure third-party library APIs `boost::get` and difficulty in debugging due to exceptions during running, adds the `BOOST_GET` series of macros to replace over 600 risky `boost::get` in Paddle. Richens error message during `boost::bad_get` exceptions. Specifically, adds the C++ error message stack, error file and line No., expected output type, and actual type, thus improving the debugging experience

## Bug Fixes

- Fix the bug of wrong computation results when any slice operation exists in the while loop
- Fix the problem of degradation of the transformer model caused by inplace ops
- Fix the problem of running failure of the last batch in the Ernie precision test
- Fix the problem of failure to correctly exit when exceptions occur in context of fluid.dygraph.guard
