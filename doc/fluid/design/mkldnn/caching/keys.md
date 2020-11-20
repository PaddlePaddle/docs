# Design Doc: MKL-DNN Data Caching Keys

[Caching](./caching.md) is a mechanism using associative objects so there is a key and stored value. This document describe ideas behind
construction of keys used for caching oneDNN objects. Maintaining of keys is an ongoing effort so they tend to evolve.

Key may consists of a several elements:
* Output tensor name
* Inplace information
* Input shape
* Executor identification
* Thread information

### 1. Output tensor name
The foundation of generation of key is name of output Tensor of given instance of operator. The reason for that is operators in PaddlePaddle are stateless. So for example
the same kernel implementing convolution operation will be used regardless the shapes and attributes of convolution. In practice this kernel for diffrent shapes, attributes, data format may require
diffrent oneDNN convolution primitive. As instance of convolution is identified uniquely with name of its output tensor then this name of output tensor was used as part of caching key.

### 2. Inplace information
One of situations when output tensor name is not enough to uniquely identify the instance of operator is when operators are performing [inplace](../inplace/inplace.md) execution. In this situation
one of inputs and one of outputs of operator are the same Paddle Tensor and its name is the same so it is not enough to uniquely indentify instance of operator. So for that situation
operators supprting inplace execution mode are having its caching key extended as follows:
* softmax's key is having a value of attribute of normalization axe added
* activation's key is having an enum determining type of activation added
* elementwise_mul's key is having character "M" added

### 3. input shape
Another situation is when PaddlePaddle is executing a model where the input shape can vary from iteration to iteration . This happens with NLP (BERT, ERNIE) models.
Then given operator instance may process diffrent input signal each iteration and oneDNN primitive potentialy may use diffrent implementation , so to stay on save side
we need to include input tensor shape information into caching key.

### 3. Executor identification
On a scenario when there are are multiple models executed in a interleaved mode e.g. one iteration of model A followed by one iteration of model B and then again one iteration of model A , it
is needed to also add information to they key on what model is executed to avoid sitation that weights of one model will be used for weights of other model. Model during inference is connected
to Executor or Naive Executor so part of address of used Executor or Naive Executor is made a part of key as well.

### 4. Thread information
When there is a multi-threaded execution of model e.g. Paralel Executor where there are multiple threads executing single model or number of threads where each of them is executing its own model
then it is needed to add to the information on Thread ID which is a hash of thread id (provided by C++ thread structures). This element is only aplied when cache is working in a mode where
multi threading is present.
