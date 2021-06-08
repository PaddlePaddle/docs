# Design Doc: MKL-DNN Data Caching Keys

[Caching](./caching.md) is a mechanism using associative objects so there is a key and stored value. This document describes ideas behind the
construction of keys used for caching oneDNN objects. Maintaining of keys is an ongoing effort so they tend to evolve.

Key may consist of a several elements:
* Output tensor name
* Inplace information
* Input shape
* Executor identification
* Thread information

### 1. Output tensor name
The foundation of generation of a key is the name of output Tensor of given instance of operator. The reason for that is operators in PaddlePaddle are stateless. So for example
the same kernel implementing convolution operation will be used regardless of the shapes and attributes of convolution. In fact this kernel for different shapes, attributes, data format may require
different oneDNN convolution primitive. As an instance of convolution is identified uniquely with name of its output tensor then this name of output tensor was used as part of the caching key.

### 2. Inplace information
One of situations when output tensor name is not enough to uniquely identify the instance of the operator is when operators are performing [inplace](../inplace/inplace.md) execution. In this situation
one of inputs and one of outputs of the operator are the same Paddle Tensor and its name is the same so it is not enough to uniquely identify the instance of an operator. So for that situation
operators supporting inplace execution mode are having its caching key extended as follows:
* softmax's key is having a value of normalization axe attribute added
* activation's key is having an enum determining the type of activation added
* elementwise mul's key is having character "M" added

### 3. input shape
Another situation is when PaddlePaddle is executing a model where the input shape can vary from iteration to iteration. This happens with NLP (BERT, ERNIE) models.
Then given operator instance may process different input signal each iteration and oneDNN primitive potentially may use a different implementation , so to stay on save side
we need to include input tensor shape information into the caching key.

### 3. Executor identification
On a scenario when there are multiple models executed in a interleaved mode e.g. one iteration of model A followed by one iteration of model B and then again one iteration of model A, it
is needed to also add information to the key on what model is executed to avoid a situation that weights of one model will be used for weights of other model. Model during inference is connected
to Executor or Naive Executor so part of the address of used Executor or Naive Executor is made a part of the key as well.

### 4. Thread information
Thread based information is needed as PaddlePaddle is often used in a multi-threaded execution of model e.g. Parallel Executor where there are multiple threads executing a single model or a number of threads where each of them is executing its own model. Then it is needed to add to the key an information about Thread ID which is a hash of thread id (provided by C++ thread structures). This element is only applied when cache is working in a mode where
multi threading is present.
