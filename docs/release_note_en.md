
# Release Note

## **1. Highlights**

We are excited to release the PaddlePaddle Framework V2.2.0. This version contains the following highlights.

### API

- Added 100+ APIs, including 24 Fourier transform APIs, 17 linear algebra APIs, etc., to better facilitate developing of scientific computing and signal processing models.
- Added the support for multiple indexing syntax, including ellipsis (...), dimension expansion (None), boolean arrays (Bool Mask), and integer arrays (list and tensor), making it easier to operate on tensor.
- Added the `paddle.einsum` API, to express multi-dimensional tensor computation in a more concise way.  
- Enhanced the dynamic graph mixed precision. Added a way to use half-precision (float16) training for the whole task. The computational efficiency under the main tasks increased by 20%.

### IR(Intermediate Representation)

- Dynamic graph to static graph conversion: Further expand the syntax and scenarios supported by dynamic-static conversion. Now the dynamic graph models trained with mixed precision can also be converted to static graphs for training or inference deployment via the `to_static` interface. In addition, the training performance after conversion can be optimized, and the training performance after conversion is significantly improved with the comparison to the dynamic graph method by introducing caching and enabling the Pass and other strategies.
- Pass development: Added the interface for rewriting static graph IR in Python, so that development can be completed quickly in python for OP fusion and other subgraph replacement scenarios.
- Abstraction and functional encapsulation of the underlying codes in the operator Kernel: Provide high-performance Block-level IO operations and Compute operations (Kernel Primitive API).The Kernel development using the Kernel Primitive API allows you to focus more on the implementation of the computational logic, significantly reducing the amount of codes while ensuring performance, and decoupling operator computation from hardware.

### **Distributed**

- Hybrid parallel: Based on the existing 4D hybrid parallel of static graph, the performance optimization such as pipeline executor is carried out, and the training arithmetic utilization reaches 51% of the theoretical peak performance of GPU under 100 billion models. The dynamic graph supports 4D hybrid parallelism, and the function and performance under 100 billion models are the same as static graphs. The basic functions such as auto-completion and auto-slicing are added, and semi-automatic parallelism based on user mark is available.
- GPU Parameter Server: Under the 100 billion models, optimize the data reading, GPU-PS construction, SSD performance, and improve the pipeline. The overall performance is doubled and memory usage is halved, and one GPU machine can replace one hundred CPU machines to train 100 billion models.

### **Inference engine**

- Inference acceleration: Support the latest TensorRT 8.x, and adapt Nvidia's new hardware features for acceleration.
- Ease of Inference: Add automatic derivation of dynamic Shape configurations in TensorRT subgraphs. Optionally, derive the range of Shapes from data without trivial manual configuration. This can simplify the use of dynamic Shape.


## **2. Backwards Incompatible changes**

- For the problem of `grad` being exposed in paths (`paddle.autograd,grad`, `paddle.grad`), it is recommended to use `paddle.grad` , with removing `from paddle.autograd import *` and calling the grad directly. ([#35579](https://github.com/PaddlePaddle/Paddle/pull/35579))

<table>
<tr>
<th>
2.1
</th>
<th>
2.2
</th>
</tr>

<tr>
<td>
<pre>

```python
>>> import paddle
>>> from paddle.autograd import *
>>> x = paddle.ones(shape=[1], dtype='float32')
>>> x.stop_gradient = False
>>> y = x*x
>>> grad(outputs=[y], inputs=[x])
[Tensor(shape=[1], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
        [2.])]
```
</pre>
</td>

<td>
<pre>

```python
>>> import paddle
>>> from paddle.autograd import *
>>> x = paddle.ones(shape=[1], dtype='float32')
>>> x.stop_gradient = False
>>> y = x*x
>>> grad(outputs=[y], inputs=[x])
NameError: name 'grad' is not defined
>>> paddle.grad(outputs=[y], inputs=[x]) # 改用paddle.grad API
[Tensor(shape=[1], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
       [2.])]
```
</pre>
</td>
</tr>
</table>

- ``Tensor.__setitem__`` does not support the slice index of non- ``int`` type ( ``x[start:stop:step] = value`` ). Since the ``float`` type does not make mathematical sense when used as an index (For example, how to determine the exact index position when ``start`` is 0.5?) and it is prone to some unknown behaviors, we limit the data type of slice index to ``int`` in this update, and the slice index using ``float`` will report an error. ([#35701](https://github.com/PaddlePaddle/Paddle/pull/35701))

<table>
<tr>
<th>
2.1
</th>
<th>
2.2
</th>
</tr>

<tr>
<td>
<pre>

```python
>>> import paddle
>>> x = paddle.to_tensor([1, 2, 3, 4])
>>> start = paddle.zeros([1])
>>> stop = paddle.zeros([1]) + 2
>>> step = paddle.ones([1])
>>> x[start:stop:step] = 0 # start,stop,step supports the float type Tensor
>>> x 
```
</pre>
</td>

<td>
<pre>

```python
>>> import paddle
>>> x = paddle.to_tensor([1, 2, 3, 4])
>>> start = paddle.zeros([1])
>>> stop = paddle.zeros([1]) + 2
>>> step = paddle.ones([1])
>>> x[start:stop:step] = 0
ValueError: (InvalidArgument) Currently, the type of tensor in slice indices only allows int32 and int64, please check the type of index tensor.

>>> # Must be changed to the following codes:
>>> start = paddle.zeros([1], dtype='int32')
>>> stop = paddle.zeros([1], dtype='int32') + 2
>>> step = paddle.ones([1], dtype='int32')
>>> x[start:stop:step] = 0 # start,stop,step must be integer type Tensor
>>> x
Tensor(shape=[4], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
       [0, 0, 3, 4])
```
</pre>
</td>
</tr>
</table>


- Add inplace to call legality check for dynamic graph ``Tensor.__setitem__``. When the detected assignment code is not met, an error will be reported (detection logic: when ``Tensor`` is a leaf node and ``stop_gradient`` is ``False``, the ``Tensor`` assignment operation will be intercepted with reporting an error).Since the execution of ``tensor[index]=value`` will overwrite the original value of the ``Tensor``, it is an inplace operation of the ``Tensor``. If the ``Tensor`` is a leaf node in the computation graph and needs to calculate the gradient, the assignment of the ``Tensor`` will cause problems in the calculation of the inverse gradient of the ``Tensor``, which is an illegal inplace operation. Therefore, we add the detection and interception of such operations in this update. For the current code with the assignment by using ``tensor [index]=value``, check whether the inplace operation requirement is met. If it is not met, an error is reported.  ([#35701](https://github.com/PaddlePaddle/Paddle/pull/35701))
  - Example: The initialization code is adjusted by using ``weight[index]=value``. The ``self.weight`` belongs to the leaf node and needs to calculate the gradient, so the inplace operation cannot be used (it will affect the inverse gradient value calculation). However, the initialization assignment itself does not need the inverse calculation process. Therefore, use ``no_ grad`` to disable the gradient calculation and then assign the value when it is clear that the inverse calculation is not needed.

<table>
<tr>
<th>
2.1
</th>
<th>
2.2
</th>
</tr>

<tr>
<td>
<pre>

```python
>>> import paddle
>>> class MyLayer(paddle.nn.Layer):
...     def __init__(self):
...         super(MyLayer, self).__init__()
...         self.weight = self.create_parameter(...)
...         self.weight[index] = 1.0
...
```
</pre>
</td>

<td>
<pre>

```python
>>> import paddle
class MyLayer(paddle.nn.Layer):
...     def __init__(self):
...         super(MyLayer, self).__init__()
...         self.weight = self.create_parameter(...)
...         with paddle.no_grad(): # Assignment can be done after gradient calculation is disabled.
...             self.weight[index] = 1.0
```
</pre>
</td>
</tr>
</table>


- When the `paddle.sum` input type is ``bool``, the output type is also bool, and the action is not consistent with ``numpy.sum``. To solve the problem, upgrade the incompatibility. After the upgrade, the output type is ``int64``, which is consistent with ``numpy.sum``. ([#34313](https://github.com/PaddlePaddle/Paddle/pull/34313))


<table>
<tr>
<th>
2.1
</th>
<th>
2.2
</th>
</tr>

<tr>
<td>
<pre>

```python
>>> import paddle
>>> import numpy as np
>>> np_arr = np.ones((2, 3), dtype='bool')
>>> pd_arr = paddle.to_tensor(np_arr)
>>> pd_sum = pd_arr.sum(0)
>>> pd_sum.dtype
paddle.bool
```
</pre>
</td>

<td>
<pre>

```python
>>> import paddle
>>> import numpy as np
>>> np_arr = np.ones((2, 3), dtype='bool')
>>> pd_arr = paddle.to_tensor(np_arr)
>>> pd_sum = pd_arr.sum(0)
>>> pd_sum.dtype
paddle.int64
```
</pre>
</td>
</tr>
</table>

- Optimize the ``Tensor`` copying act in the case where ``paddle.to_tensor`` does not copy the ``Tensor`` when the input ``data`` is a ``Tensor``, causing the ``stop_gradient`` property to be incorrectly modified. In the original implementation, when ``data`` is a ``Tensor`` and ``dtype`` and ``place`` do not change, ``data`` is returned directly (i.e., no copying occurs) and the ``data.stop_gradient`` property is modified. This action will cause the problem of the back propagation of the original computed graph ``data``. In the new implementation, the ``paddle.to_tensor`` copies a new ``Tensor`` and returns it in the above case, without modifying the ``stop_gradient`` property of the original ``data``.  ([#33335](https://github.com/PaddlePaddle/Paddle/pull/33335)) 

<table>
<tr>
<th>
2.1
</th>
<th>
2.2
</th>
</tr>

<tr>
<td>
<pre>

```python
>>> import paddle
>>> x = paddle.rand([2,3])
>>> x.stop_gradient = False
>>> y = paddle.to_tensor(x)
>>> print(id(x) == id(y)) # True
>>> print(x.stop_gradient, y.stop_gradient) # True True
```
</pre>
</td>

<td>
<pre>

```python
>>> import paddle
>>> x = paddle.rand([2,3])
>>> x.stop_gradient = False
>>> y = paddle.to_tensor(x)
>>> print(id(x) == id(y)) # False
>>> print(x.stop_gradient, y.stop_gradient) # False True
```
</pre>
</td>
</tr>
</table>

## **3. Training framework (with distributed)**

### **(1) New features**

#### API

- Add the linear algebra computation API  ``paddle.linalg.*``
 - Add the ``paddle. linalg.svd``, to support the singular value decomposition for multi-dimensional ``Tensor``.  ([#34953](https://github.com/PaddlePaddle/Paddle/pull/34953)) 
   - Add the ``paddle.linalg.cond``, to support the computing of the condition number of a matrix or a batch of matrixes based on the norm type ``p``.  ([#35140](https://github.com/PaddlePaddle/Paddle/pull/35140)) 
   - Add the ``paddle.linalg.matrix_rank``, to support the computing of the rank of a multidimensional matrix ``Tensor``.  ([#34823](https://github.com/PaddlePaddle/Paddle/pull/34823)) 
   - Add the ``paddle.linalg.eigvals``, to support the computing of general squares.  ([#35720](https://github.com/PaddlePaddle/Paddle/pull/35720), [#35909](https://github.com/PaddlePaddle/Paddle/pull/35720))
   - Add the ``padding.linalg.eigh``, to support the computing of eigenvalues and eigenvectors of complex Hermite matrix or real symmetric matrix. ([#34990](https://github.com/PaddlePaddle/Paddle/pull/34990), [#35916](https://github.com/PaddlePaddle/Paddle/pull/35916), [#35812](https://github.com/PaddlePaddle/Paddle/pull/35812), [#36091](https://github.com/PaddlePaddle/Paddle/pull/36091),[#35919](https://github.com/PaddlePaddle/Paddle/pull/35919)) 
   - Add the ``paddle.linalg.det``, to support the computing of determinant values of multidimensional matrix.  ([#34992](https://github.com/PaddlePaddle/Paddle/pull/34992)) 
   - Add the ``paddle.linalg.slogdet``, to support the computing of signed and natural logarithm values of multidimensional matrix determinant values. ([#34992](https://github.com/PaddlePaddle/Paddle/pull/34992))
   - Add the ``paddle.linalg.pinv``, to support the computing of pseudo-inverse matrix of multidimensional matrix Tensor.  ([#35804](https://github.com/PaddlePaddle/Paddle/pull/35804))
   - Add the ``paddle.linalg.multi_dot``, to support the computing of concatenated multiplication of multiple matrices. ([#35224](https://github.com/PaddlePaddle/Paddle/pull/35224))
   - Add the ``paddle.linalg.solve``, to support the computing of the solutions of linear equations.  ([#35715](https://github.com/PaddlePaddle/Paddle/pull/35715))
   - Add the ``paddle.linalg.matrix_power``, to support the power operations on matrices.  ([#34667](https://github.com/PaddlePaddle/Paddle/pull/34667))
   - Add `paddle.linalg.eigvalsh` for computing eigenvalues of Hermite Matrix or real symmetric matrices.  ([#36680](https://github.com/PaddlePaddle/Paddle/pull/36680))
   - Add `paddle.linalg.eig` for computing eigenvalues and eigenvectors of general square matrices.  ([#35674](https://github.com/PaddlePaddle/Paddle/pull/35674))
   - Add `paddle.linalg.qr` for computing QR decomposition of matrices (inverse is not supported yet).  ([#36627](https://github.com/PaddlePaddle/Paddle/pull/36627))

- Add new Fourier transform related API ([#35665](https://github.com/PaddlePaddle/Paddle/pull/35665)) 
  - Add fast Fourier transform family functions  
    - Differentiable 1d to nd complex to complex fast Fourier transforms.  (``paddle.fft.fft``, ``paddle.fft.fft2``, ``paddle.fft.fftn``, ``paddle.fft.ifft``, ``paddle.fft.ifft2``, ``paddle.fft.ifftn``)
    - Differentiable 1d to nd real to complex fast Fourier transform.  (``paddle.fft.rfft``, ``paddle.fft.rfft2``, ``paddle.fft.rfftn``, ``paddle.fft.ihfft``, ``paddle.fft.ihfft2``, ``paddle.fft.ihfftn``)
    - Differentiable 1d to nd complex to real fast Fourier transform.  (``paddle.fft.hfft``, ``paddle.fft.hfft2``, ``paddle.fft.hfftn``, ``paddle.fft.irfft``, ``paddle.fft.irfft2``, ``paddle.fft.irfftn``)
    - fft related helper functions. (``paddle.fft.fftfreq``, ``paddle.fft.rfftfreq``, ``paddle.fft.fftshift``, ``paddle.fft.ifftshift``)

  - Add short-time Fourier transform related functions
    - Short-time Fourier transform.  (``paddle.signal.stft``)
    - Short-time Fourier inverse transform.  (``paddle.signal.istft``)

- Add new high-level APIs  
  - Add the ``paddle.vision.ops.roi_pool`` and ``paddle.vision.ops.RoIPool``, support RoI region pooling operations in detection tasks. ([#36154](https://github.com/PaddlePaddle/Paddle/pull/36154))
    -  Add the ``paddle.vision.ops.roi_align`` and ``paddle.vision.ops.RoIAlign``, to support RoI region Align operations in detection tasks.  ([#36207](https://github.com/PaddlePaddle/Paddle/pull/36207))
    -  Add the ``paddle.vision.ops.psroi_pool`` and ``paddle.vision.ops.PSRoIPool``, to support location-sensitive RoI region pooling operations in detection tasks. ([#36111](https://github.com/PaddlePaddle/Paddle/pull/36111))
    -  Add the ``paddle.vision.models.vgg19`` pre-training weights. ([#35788](https://github.com/PaddlePaddle/Paddle/pull/35788))
    -  Add thedatasets API download progress bar in ``paddle.vision.datasets.*``. ([#33302](https://github.com/PaddlePaddle/Paddle/pull/33302))
    -  Add the ``paddle.Model.predict`` parameter ``verbose``, to support whether to show logs or not. ([#33405](https://github.com/PaddlePaddle/Paddle/pull/33405))
    -  Add the ``paddle.hub`` download option ``wget`` method. ([#33379](https://github.com/PaddlePaddle/Paddle/pull/33379))
    -  Add the ``paddle.Model`` gradient accumulation in dynamic graph mode. ([#32702](https://github.com/PaddlePaddle/Paddle/pull/32702))
    -  Add the ``paddle.Model.fit`` and ``paddle.Model.evaluate`` ``num_iters`` parameters in dynamic graph mode to control the number of training iterations. ([#33986](https://github.com/PaddlePaddle/Paddle/pull/33986))
    -  Add the ``paddle.vision.ops.yolo_box`` parameters ``iou_aware`` and ``iou_aware_factor``, to support YoloBox using predicted IOUs as confidence factors. ([#33400](https://github.com/PaddlePaddle/Paddle/pull/33400))
    -  Add the ``paddle.summary`` parameter input to support the given ``input``. ([#34165](https://github.com/PaddlePaddle/Paddle/pull/34165))
    - Add `paddle.text.viterbi_decode`, to support Viterbi decoding for CPU and GPU under dynamic graphs.  ([#35778](https://github.com/PaddlePaddle/Paddle/pull/35778))

- Add networking class APIs  
  - Add `paddle.nn.functional.sparse_attention` for computing sparse Transformer Attention modules.  ([#35757](https://github.com/PaddlePaddle/Paddle/pull/35757))
  - Add the ``paddle.nn.MaxUnPool2D`` and ``paddle.nn.functional.max_unpool2d``, to support the computing of the inverse of the pooling result based on the input and maximum position.  ([#35056](https://github.com/PaddlePaddle/Paddle/pull/35056))
  - Add the ``paddle.nn.functional.gumbel_softmax``, to support ``gumbel softmax`` sampling.  ([#35506](https://github.com/PaddlePaddle/Paddle/pull/35506), [#36065](https://github.com/PaddlePaddle/Paddle/pull/36065), [#36094](https://github.com/PaddlePaddle/Paddle/pull/36094))
  - Add the ``paddle.nn.functional.class_center_sample``, to support PartialFC class center sampling. ([#34106](https://github.com/PaddlePaddle/Paddle/pull/34106))
  - Add the ``paddle.nn.functional.margin_cross_entropy``, to support ArcFace, CosFace, SphereFace and other MarginLoss functions. ([#34247](https://github.com/PaddlePaddle/Paddle/pull/34247))
  - Add the ``paddle.nn.AvgPool2D``, to support second-order derivatives. ([#35388](https://github.com/PaddlePaddle/Paddle/pull/35388))
  - Add the ``paddle.nn.Linear, paddle.matmul, and paddle.mm``, to support second-order derivatives.  [#35428](https://github.com/PaddlePaddle/Paddle/pull/35428)
  - Add the ``paddle.nn.GroupNorm``, to support the inputs of the form (N, C, *). ([#34773](https://github.com/PaddlePaddle/Paddle/pull/34773))
  - Add the ``paddle.nn.BatchNorm1D/2D/3D`` to compute the inverse under ``x.stop_gradient=True``. ([#34102](https://github.com/PaddlePaddle/Paddle/pull/34102))
  - Add the ``paddle.nn.Dropout, paddle,nn.Dropout2D/3D`` to compute the inverse in ``model.eval`` mode.  ([#35122](https://github.com/PaddlePaddle/Paddle/pull/35122))

- Add hardware related APIs  
  - Add the `paddle.device.cuda.Stream`, `paddle.device.cuda.Event`, `paddle.device.cuda.current_stream`, `paddle.device.cuda.synchronize`, `paddle.device.cuda.synchronize`, to support synchronization operations for event and stream of CUDA on the Python side. ([#32460](https://github.com/PaddlePaddle/Paddle/pull/32460))
  - Add the ``paddle.device.cuda.device_count``, to support returning the current number of available GPUs. ([#34811](https://github.com/PaddlePaddle/Paddle/pull/34811))
  - Add the ``paddle.device.cuda.empty_cache``, to support for clearing free GPU memory.  ([#35427](https://github.com/PaddlePaddle/Paddle/pull/35427))
  - Add the ``paddle.device.cuda.get_device_properties``, to support for returning the given device properties. ([#35875](https://github.com/PaddlePaddle/Paddle/pull/35875))
  - Add the ``paddle.device.cuda.stream_guard`` for flexible switching of CUDA Streams under dynamic graphs. ([#35623](https://github.com/PaddlePaddle/Paddle/pull/35623))
  - Add `paddle.device.cuda.get_device_name`, to support returning the name of a given device.  ([#36172](https://github.com/PaddlePaddle/Paddle/pull/36172))
  - Add `paddle.device.cuda.get_device_capability`, to support returning version number of the computational capability of a given device.  ([#36172](https://github.com/PaddlePaddle/Paddle/pull/36172))
  - Add `paddle.framework.core.async_read` and `paddle.framework.core.async_write`, to support `Tensor` data asynchronous read and write of `CUDAPinnedPlace` and ` CUDAPlace` under non-default CUDA `Stream`.  ([#36501](https://github.com/PaddlePaddle/Paddle/pull/36501))


- Add Tensor operation APIs  
 - Add `paddle.tensordot`, to support Tensor Contraction for high dimension.  ([#36454](https://github.com/PaddlePaddle/Paddle/pull/36454))
 - Add `paddle.bincount`, to support counting elements in a one-dimensional tensor.  ([#36709](https://github.com/PaddlePaddle/Paddle/pull/36709))
 - Add the `paddle.broadcast_tensors`, to support broadcast operations on a set of `Tensors`.  ([#33294](https://github.com/PaddlePaddle/Paddle/pull/33294), [#34874](https://github.com/PaddlePaddle/Paddle/pull/34874))
 - Add the `paddle.einsum`.  ([#33821](https://github.com/PaddlePaddle/Paddle/pull/34874))
 - Enhance the ``paddle.tensor.gradient`` interface to support second-order derivative operators for sigmoid_op. ([#32971](https://github.com/PaddlePaddle/Paddle/pull/32971))
 - Add the ``paddle.searchsorted``, to support the search of the index of a given value in an ordered ``Tensor``.  ([#35159](https://github.com/PaddlePaddle/Paddle/pull/35159))
 - Add the ``paddle.unique_consecutive``, to support removing duplicates of consecutively repeated elements in a ``Tensor`` to return consecutive non-repeated Tensor. ([#34334](https://github.com/PaddlePaddle/Paddle/pull/34334))
 - Add the ``paddle.diagflat``, to support the returning of a diagonal matrix with the elements of the input ``Tensor`` as diagonals.  ([#33334](https://github.com/PaddlePaddle/Paddle/pull/33334))
 - Add the ``paddle.lgamma``, to support element-by-element computing of the ``Tensor``'s ``lgamma`` function value.  ([#33913](https://github.com/PaddlePaddle/Paddle/pull/32913))
 - Add the ``paddle.digamma``, to support element-by-element computing of ``digamma`` function values for ``Tensor``. ([#33278](https://github.com/PaddlePaddle/Paddle/pull/33278))
 - Add the ``paddle.neg``, to support element-by-element computing of the opposite value of a ``Tensor``.  ([#33248](https://github.com/PaddlePaddle/Paddle/pull/33248))
 - Add the ``paddle.cumprod``, to support the computing of ``Tensor`` cumulative multiplication based on a given dimension. ([#35185](https://github.com/PaddlePaddle/Paddle/pull/35185))
 - Add the ``paddle.atan2``, to support element-by-element ``arctangent`` operations to determine quadrants by symbols. ([#33067](https://github.com/PaddlePaddle/Paddle/pull/33067))
 - Add the ``paddle.expm1``, to support element-by-element arithmetic with ``exp(x)-1``.  ([#33066](https://github.com/PaddlePaddle/Paddle/pull/33066))
 - Add the ``paddle.trunc``, to support truncated integer values for the input ``Tensor``. ([#33371](https://github.com/PaddlePaddle/Paddle/pull/33371))
 - Add the ``paddle.diagonal``, to support the extracting of diagonal elements of the input ``Tensor``.  ([#33586](https://github.com/PaddlePaddle/Paddle/pull/33586)) 
 - Add the ``paddle.utils.dlpack``, including: ``paddle.utils.dlpack.to_dlpack`` and ``paddle.utils.dlpack.from_dlpack``, to support the ``Tensor`` transfer between different frameworks with using ``DLPack``.  ([#35067](https://github.com/PaddlePaddle/Paddle/pull/35067))
 - Add the ``Tensor.uniform``_, to support filling a ``Tensor`` in-place with random numbers that obey a uniform distribution. ([#33394](https://github.com/PaddlePaddle/Paddle/pull/33934))
 - Add the ``paddle.Tensor.T``, to transpose an N-D Tensor to return a Tensor with the opposite shape of the original Tensor.  ([#35379](https://github.com/PaddlePaddle/Paddle/pull/35379)) 
 - Add the ``paddle.Tensor`` magic operators: & (bitwise_and), | (bitwise_or), ^ (bitwise_xor), ~ (bitwise_not). ([#33524](https://github.com/PaddlePaddle/Paddle/pull/33524))
 - Add the `paddle.Tensor.fill_ `and `paddle.Tensor.zero_`, to modify the value in Tensor in-place, use the fixed values to fill, use all-zero to fill respectively. ([#33829](https://github.com/PaddlePaddle/Paddle/pull/33829)) 
 - Add the `paddle.Tensor.fill_diagonal`, and `paddle.Tensor.fill_diagonal`, to modify Tensor diagonal element values. ([#34460](https://github.com/PaddlePaddle/Paddle/pull/34460)) 
 - Add the `paddle.Tensor.fill_diagonal_tensor_`, to modify the whole sub-Tensor formed by the diagonal of two specified coordinate axes of the Tensor with other axes. ([#34515](https://github.com/PaddlePaddle/Paddle/pull/34515)) 
 - Dynamic-Static Graph ``Tensor``: Add the support for multiple index types, including: ellipsis (...), dimensional augmentation (None), boolean type arrays (Bool Mask), integer arrays (list), and tensors (Tensor).  
   - ellipsis (...) Index:  `X[..., 0]` 。([#34267](https://github.com/PaddlePaddle/Paddle/pull/34267), [#32876](https://github.com/PaddlePaddle/Paddle/pull/32876))
   - Dimensional augmentation (None) index:   `X[None, :]` 。([#34338](https://github.com/PaddlePaddle/Paddle/pull/34338), [#34442](https://github.com/PaddlePaddle/Paddle/pull/34442),  [#34877](https://github.com/PaddlePaddle/Paddle/pull/34877),  [#34911](https://github.com/PaddlePaddle/Paddle/pull/34911),  [#33001](https://github.com/PaddlePaddle/Paddle/pull/33001))
    - Boolean type array (Bool Mask) index:  `X[X > 0] = 0` 。 ([#35026](https://github.com/PaddlePaddle/Paddle/pull/35026),  [#35133](https://github.com/PaddlePaddle/Paddle/pull/35133),  [#33298](https://github.com/PaddlePaddle/Paddle/pull/33298))
    - Array of integers (list) index: `X[[1, 0], [0]]` 。([#34824](https://github.com/PaddlePaddle/Paddle/pull/34824), [#33000](https://github.com/PaddlePaddle/Paddle/pull/33000),  [#35404](https://github.com/PaddlePaddle/Paddle/pull/35404))
    - Tensor index:  `X[panddle.to_tensor([0, 1], [1, 0])]` 。([#34824](https://github.com/PaddlePaddle/Paddle/pull/34824))

- Add the distributed related APIs  
  - Add the ``paddle.distributed.utils.global_scatter`` and ``paddle.distributed.utils.global_gather``, to support MOE conditional distribution of data. The ``global_scatter`` distributes the data to all cards based on the conditions, and then the ``global_gather`` then collects the data from all GPU cards based on the conditions. ([#35546](https://github.com/PaddlePaddle/Paddle/pull/35546))

- Add additional APIs  
  -  Add the ``paddle.disable_signal_handler``, to support the disabling of the signal capture mechanism in PaddlePaddle, thus allow users to use Paddle and TVM at the same time. ([#34577](https://github.com/PaddlePaddle/Paddle/pull/34577))
  -  Add the ``paddle.incubate.softmax_mask_fuse``, to support the acceleration of softmax and mask operations for Transformer architecture. ([#33841](https://github.com/PaddlePaddle/Paddle/pull/33841))
  -  Add the ``paddle.incubate.softmax_mask_fuse_upper_triangle``, to support the acceleration of the softmax and mask operations of the GPT version of the Transformer architecture. ([#33981](https://github.com/PaddlePaddle/Paddle/pull/33981))
  -  Add the ``paddle.static.ExponentialMovingAverage``, to support the computing of the sliding average of parameters with exponential decay. ([#35673](https://github.com/PaddlePaddle/Paddle/pull/35673))
  -  Add the ``paddle::Tensor::slice`` C++ API, to support the slice operation, and allow users to perform slice operations for the external Tensor. ([#34227](https://github.com/PaddlePaddle/Paddle/pull/34227))
  -  Add the ``paddle.incubate.segment_*`` series APIs, including ``paddle.incubate.segment_sum``, ``paddle.incubate.segment_mean``, ``paddle.incubate.segment_max``, and ``paddle. incubate.segment_min``. Support the summing, averaging, maximizing, and minimizing of ``Tensor`` by segment.  ([#35759](https://github.com/PaddlePaddle/Paddle/pull/35759))
  - Add `paddle.version.cuda` and `paddle.version.cudnn` to get version numbers of `CUDA` and `cuDNN` used by paddle installer.  ([#36556](https://github.com/PaddlePaddle/Paddle/pull/36556))


#### IR(Intermediate Representation)

- Dynamic graph to static graph 
  - Add the dynamic to static transcription error type recognition, and give suggestions for modification. ([#35648](https://github.com/PaddlePaddle/Paddle/pull/35648)) 
  - Add the support for mixed precision training. ``@to_static`` c supports one-click conversion to mixed precision training mode for static graphs.  ([#34562](https://github.com/PaddlePaddle/Paddle/pull/34562))
  - Add the ``build_strategy`` parameter in ``@to_static``. Support customizing the ``Pass`` optimization strategy to accelerate model training after dynamic to static, such as operator fusion, etc.   ([#34347](https://github.com/PaddlePaddle/Paddle/pull/34347))
  - Add the support for `a, b = static_variable`.  ([#33499](https://github.com/PaddlePaddle/Paddle/pull/33499))
  - Add the support for second-order derivatives.  ([#33110](https://github.com/PaddlePaddle/Paddle/pull/33110))

- Program and Graph conversion: ``Program`` and ``Graph`` are the intermediate representations used to express computations in the underlying framework of the PaddlePaddle, or developers of the PaddlePaddle, it is sometimes necessary to convert ``Program`` and ``Graph`` to each other for computational processing. This feature adds the ability to convert ``Program`` and ``Graph`` to each other.  
  - Develop and refine the ``Program`` and ``Graph`` interconversion feature.  ([#33949](https://github.com/PaddlePaddle/Paddle/pull/33949))
  - In order to support control flow nodes such as `while`, the `Program` of the PaddlePaddle Framework may contain multiple sub-`blocks` in addition to the main `block`. Previously, in the conversion of `Program` to `Graph`, only convert the main `block` to `Graph`. In this update, modify the `Graph`, to support the expression of sub-`blocks` to achieve a complete conversion of `Program` to `Graph`.  ([#33320](https://github.com/PaddlePaddle/Paddle/pull/33320))
  - Provide dependent helper functions needed to analyze the control flow in `Program`.  ([#33439](https://github.com/PaddlePaddle/Paddle/pull/33439))
  - `Program` and `Graph` retain the values of the `stop_gradient` and `persistable` attributes needed for training after converting each other.  ([#33771](https://github.com/PaddlePaddle/Paddle/pull/33771)) 
  - `Pass` now supports processing the main `Graph` and all its sub-graphs, while the original `Pass` only processed the main `Graph` and ignored the sub-graphs.  ([#34158](https://github.com/PaddlePaddle/Paddle/pull/34158)) 
  - Handle some topological ordering problems for `Program` and `Graph` inter-conversion in the prediction cases. ([#34121](https://github.com/PaddlePaddle/Paddle/pull/34121), [#34521](https://github.com/PaddlePaddle/Paddle/pull/34521)).

- Pass development  
  - Add the Pass development for subgraph replacement scenarios such as fusion on the Python side.  ([#35708](https://github.com/PaddlePaddle/Paddle/pull/35708), [#35602](https://github.com/PaddlePaddle/Paddle/pull/35602))

- Kernel Primitive API	
  - Abstract and encapsulate the underlying codes in the operator Kernel implementation, to provide high-performance Block-level IO and Compute operations. The Kernel development using the Kernel Primitive API allows you to focus more on the implementation of the computational logic, significantly reducing the amount of codes while ensuring performance, and decoupling operator computation from hardware.  ([#34672](https://github.com/PaddlePaddle/Paddle/pull/34672),  [#35075](https://github.com/PaddlePaddle/Paddle/pull/35075),  [#34456](https://github.com/PaddlePaddle/Paddle/pull/34456),  [#35282](https://github.com/PaddlePaddle/Paddle/pull/35282),  [#35743](https://github.com/PaddlePaddle/Paddle/pull/35743),  [#34208](https://github.com/PaddlePaddle/Paddle/pull/34208))
  - Add a total of 13 monadic and binary computation Functors to the Kernel Primitive API.  ([#36418](https://github.com/PaddlePaddle/Paddle/pull/36418))
  - Modify the ReadData implementation in the Kernel Primitive API to fix the NX ! =1 access memory out-of-bound bug.  ([#36373](https://github.com/PaddlePaddle/Paddle/pull/36373))

#### **Mixed Precision Training**

- Enhance the dynamic graph mixed precision. Add a way to use half-precision (float16) training for the whole task. The computational efficiency under the main task increases by 20%. ([#35521](https://github.com/PaddlePaddle/Paddle/pull/35521))
- In the dynamic graph mixed precision ``paddle.amp.GradScaler``, add the ``get`` and ``set`` methods for user-friendly settings. ([#33835](https://github.com/PaddlePaddle/Paddle/pull/33835))
- In the dynamic graph mixed precision ``paddle.amp.GradScaler``, add the ``state_dict`` and ``load_state_dict`` methods.  ([#34300](https://github.com/PaddlePaddle/Paddle/pull/34300))
- In the dynamic graph mixed precision, split ``minimize`` to ``step + update``. In addition, add the ``unscale`` method.  ([#35927](https://github.com/PaddlePaddle/Paddle/pull/35927))
- In the dynamic graph mixed precision training, support param group. ([#34899](https://github.com/PaddlePaddle/Paddle/pull/34899))
- In the static graph mixed precision training, support the gradient pruning.   ([#33565](https://github.com/PaddlePaddle/Paddle/pull/33565))


#### **Distributed training**

- Basic functions of distributed training
  - Add `paddle.DataParallel.no_sync`, to pause multi-card communication and gradient synchronization under dynamic graph data parallelism.  ([#34740](https://github.com/PaddlePaddle/Paddle/pull/34740)) 
  - Add the `paddle.distributed.launch`, to start the mode support for fault tolerance, and implement fault tolerance for nodes in `collective` mode.  ([#33369](https://github.com/PaddlePaddle/Paddle/pull/33369),  [#34572](https://github.com/PaddlePaddle/Paddle/pull/34572))
  - In the distributed training API `paddle.static.Executor.train_from_dataset`, `paddle.static.Executor.infer_from_dataset`, add the dump function for parameters and intermediate variables of the model during training. [#34457](https://github.com/PaddlePaddle/Paddle/pull/34457) 
  - In the hybrid parallel, support the combination of model parallel and data parallel. ([#34377](https://github.com/PaddlePaddle/Paddle/pull/34377))
  - Add the distributed policy `gradient scale` option. Users can specify the way of `gradient scale`: `avg`, `sum` or custom. ([#33862](https://github.com/PaddlePaddle/Paddle/pull/33862))
  - Add `paddle.distributed.parallel_with_gloo`, support CPU barrier operation.  ([#34671](https://github.com/PaddlePaddle/Paddle/pull/34671))
  - For the GPU parameter servers add the training profiler function. ([#32640](https://github.com/PaddlePaddle/Paddle/pull/32640))
  - For the GPU parameter server, add the pipeline function. The training performance can increase by 40%.  [#33159](https://github.com/PaddlePaddle/Paddle/pull/33159)  
  - For the static graph hybrid parallel, add the `dp_as_optimizer_sharding` experimental feature that can parallelize data as optimizer parameter sharding. This can save the optimizer state GPU memory usage. ([#35593](https://github.com/PaddlePaddle/Paddle/pull/35593))
  - For the static graph pipeline parallel executor, support the `LRScheduler`.  ([#34402](https://github.com/PaddlePaddle/Paddle/pull/34402))
  - Add the `paddle.fluid.core.GraphPyClient.set_node_feat`, to support for setting graph node features in the graph engine client, support the storage of multiple types of features.  ([#34994](https://github.com/PaddlePaddle/Paddle/pull/34994))
  - Improve the performance of the graph engine graph node neighbor sampling algorithm, and optimize the execution of the graph wandering algorithm. ([#34088](https://github.com/PaddlePaddle/Paddle/pull/34088))
  - Implement the unified dynamic-static mode for the model parallel interfaces `paddle.distributed.fleet.meta_parallel.ColumnParallelLinear`, `paddle.distributed.fleet.meta_parallel.RowParallelLinear`, `paddle.distributed.fleet.meta_parallel.VocabParallelEmbedding`, and `paddle.distributed.fleet.meta_parallel.ParallelCrossEntropy`.  ([#33700](https://github.com/PaddlePaddle/Paddle/pull/33700),  [#33411](https://github.com/PaddlePaddle/Paddle/pull/33411))
  - Add the distributed model parallel `cpu c_embedding` op.  ([#35467](https://github.com/PaddlePaddle/Paddle/pull/35467))
  - Change to the retry mechanism for getting gethostbyname when gen_comm_id is added to the initialization phase of the distributed communication. ([#34855](https://github.com/PaddlePaddle/Paddle/pull/34855))
  - Add the switch configuration for `scale_sparse_gradient_with_batch_size` during `fleet` gradient update, to determine whether the gradient is multiplied by `batch_size`.   ([#34893](https://github.com/PaddlePaddle/Paddle/pull/34893))

- Dynamic graph hybrid parallel 
  - In dynamic graph distributed data parallel scenarios, add the `paddle.distributed.fleet.dygraph_optimizer.DygraphShardingOptimizer` interface. Optimize the GPU memory occupation through the sharding optimizer between cards. Support the larger model or batch size.  ([#33633](https://github.com/PaddlePaddle/Paddle/pull/33633))
  - For the dynamic graph Sharding, support the MP-PP-DP for dynamic graph 4D hybrid parallelism. ([#35580](https://github.com/PaddlePaddle/Paddle/pull/35580))
  - For the dynamic graph Recompute, support mixed precision computation. ([#33251](https://github.com/PaddlePaddle/Paddle/pull/33251))
  - For the pipeline parallel, support 1f1b scheduling policy for runtime memory savings.  ([#34483](https://github.com/PaddlePaddle/Paddle/pull/34483))
  - For the dynamic graph 3D hybrid parallel, support the recompute policy. Support the offload function.  ([#34607](https://github.com/PaddlePaddle/Paddle/pull/34607) [#35588](https://github.com/PaddlePaddle/Paddle/pull/35588))
  - For the dynamic graph 3D Hybrid Parallel, support model saving and loading.  ([#34768](https://github.com/PaddlePaddle/Paddle/pull/34768))
  - Add the scatter-gather scheme for model parallel + pipeline parallel scenarios. Optimize the cross-machine communication performance.  ([#34130](https://github.com/PaddlePaddle/Paddle/pull/34130))
  - For the pipeline parallel, support the slice based on the number of layers to ensure more equal sharding.  ([#34207](https://github.com/PaddlePaddle/Paddle/pull/34207))
  - For the pipeline parallel, support the automatic mixing precision. ([#33951](https://github.com/PaddlePaddle/Paddle/pull/33951))
  - For the pipeline parallel, add the `paddle.distributed.fleet.meta_parallel.SharedLayerDesc` the networking description, to support the parameter sharing networking mode. ([#33578](https://github.com/PaddlePaddle/Paddle/pull/33578))
  - For the tensor parallel, add `paddle.distributed.fleet.meta_parallel.ParallelCrossEntropy`, for a tensor parallel computation method that supports cross-entropy Loss.  ([#33401](https://github.com/PaddlePaddle/Paddle/pull/33401))
  - For the `paddle.DataParallel`, add the `find_unused_parameters` interface, to support the use of control flow in the model in the data parallel mode. ([#32826](https://github.com/PaddlePaddle/Paddle/pull/32826))
  - In the data parallel mode, add the port waiting feature to solve port conflict problem.  ([#34207](https://github.com/PaddlePaddle/Paddle/pull/34207))

- Static graph hybrid parallel
  - Support the fuse grad merge function under pipeline parallel. Through the `distributed_strategy.fuse_grad_merge` switch control, the performance increases by about 5%.   ([#35004](https://github.com/PaddlePaddle/Paddle/pull/35004))
  - Support the fuse allreduce sum function with enabling dp in the mixed parallel, the performance increases by 3%. ([#34480](https://github.com/PaddlePaddle/Paddle/pull/34480))

- Automatic parallel  
  - Add the auto-parallel `shard_tensor`, `shard_op` interfaces.(#33804, #35765). Support semi-automatic parallel based on user tags.
  - Add the auto-completion distributed attribute feature. Support completing all untagged distributed attributes based on user-tagged distributed attributes. ([#34813](https://github.com/PaddlePaddle/Paddle/pull/34813))
  - Add the auto-slice serial `Program` function. ([#35117](https://github.com/PaddlePaddle/Paddle/pull/35117))
  - Enable the automatic parallel adaptation of the Fleet API. ([#35483](https://github.com/PaddlePaddle/Paddle/pull/35483))


#### **Others**  

- Model quantization  
  - Add the offline quantization of dynamic graphs. ([#33445](https://github.com/PaddlePaddle/Paddle/pull/33445),  [#33898](https://github.com/PaddlePaddle/Paddle/pull/33898), [#33962](https://github.com/PaddlePaddle/Paddle/pull/33962),  [#35015](https://github.com/PaddlePaddle/Paddle/pull/35015))
  - Refactor the statistical output quantization information module in the dynamic graph quantization training function, to allow the availability on the prediction side to improve the robustness. ([#31680](https://github.com/PaddlePaddle/Paddle/pull/31680), [#31710](https://github.com/PaddlePaddle/Paddle/pull/31710), [#31861](https://github.com/PaddlePaddle/Paddle/pull/31861))
  - For the dynamic graph quantization training, support the use in combination with mixed precision training. ([#33484](https://github.com/PaddlePaddle/Paddle/pull/33484))
  - For the dynamic graph quantization training function, support the quantization of Function class API. ([#33162](https://github.com/PaddlePaddle/Paddle/pull/33162), [#33871](https://github.com/PaddlePaddle/Paddle/pull/33871))
  - Support the distributed quantization training in static graph mode. ([#33781](https://github.com/PaddlePaddle/Paddle/pull/33781))
  - Support the quantization of conv2d_transpose in dynamic graph mode. ([#34547](https://github.com/PaddlePaddle/Paddle/pull/34547))

- Custom OP
  - Add the custom operator DCU back-end support. ([#34050](https://github.com/PaddlePaddle/Paddle/pull/34050))

- Cost Model
  - Add the Paddle CostModel, to implement the method to get op time cost via Profiler.  ([#35774](https://github.com/PaddlePaddle/Paddle/pull/35774)) 

- Model saving and loading
  - Add the function of saving Layer's non-forward member methods and related parameters as inference models directly via the ``paddle.jit.save`` interface.  ([#34070](https://github.com/PaddlePaddle/Paddle/pull/34070))


- ONNX Exporter 
  - Add 8 operator adaptations: `softplus`, `elementwise_mod`, `elementwise_floordiv`, `p_norm`, `depthwise_transpose`, `group_norm`, `pixel_shuffle, top_k`. ([Paddle2ONNX#252](https://github.com/PaddlePaddle/Paddle2ONNX/pull/252),  [Paddle2ONNX#261](https://github.com/PaddlePaddle/Paddle2ONNX/pull/261),  [Paddle2ONNX#293](https://github.com/PaddlePaddle/Paddle2ONNX/pull/293))
  - Add 8 detection model exports: PPYOLO, PPYOLOv2, PPYOLO-Tiny, TTFNet, PAFNet, FCOS, SSD.   ([Paddle2ONNX#252](https://github.com/PaddlePaddle/Paddle2ONNX/pull/252))

### **(2) Function optimization**  

#### API

-   `paddle.slice`: Add the support for `bool` type Tensor and optimize error messages. ([#35586](https://github.com/PaddlePaddle/Paddle/pull/35586), [#35179](https://github.com/PaddlePaddle/Paddle/pull/35179))
-   `paddle.strided_slice`: Add the support for `TensorArray` type input, and adjust the output when `step< 0`. The adjusted result is consistent with `numpy`.  ([#34205](https://github.com/PaddlePaddle/Paddle/pull/34205), [#34172](https://github.com/PaddlePaddle/Paddle/pull/34172))
-   ``paddle.multiply``: Support ``bool`` data type operations.  ([#35551](https://github.com/PaddlePaddle/Paddle/pull/35551))
-   Logical operations (``paddle.logical_not``, ``paddle.logical_and``, ``paddle.logical_or``, ``paddle.logical_xor``): Support non-``bool`` data types (``int8, int16, int32, int64, float, double``).   ([#34141](https://github.com/PaddlePaddle/Paddle/pull/34141))
-   ``paddle.transpose``: Support ``bool`` type operations. ([#35886](https://github.com/PaddlePaddle/Paddle/pull/35886))
-   ``paddle.strided_slice``: Support ``bool`` type operations.  ([#33373](https://github.com/PaddlePaddle/Paddle/pull/33373))
-   ``paddle.set_printoptions``: Support the setting of ``linewidth`` to print ``Tensor``.  ([#35175](https://github.com/PaddlePaddle/Paddle/pull/35175))
-   ``paddle.to_tensor``: Support ``LoDTensor``.  ([#33027](https://github.com/PaddlePaddle/Paddle/pull/33027))
-   ``paddle.linalg.det`` and ``paddle.linalg.slogdet``: Support inverse operations. ([#36013](https://github.com/PaddlePaddle/Paddle/pull/36013))
-   ``paddle.nn.functional.pad``: Support the input of tuple type pad parameter in case of full dimensional pads.  ([35985](https://github.com/PaddlePaddle/Paddle/pull/35985))
-   Optimize error report messages when ``paddle.nn.functional.pad`` input is abnormal.  ([34979](https://github.com/PaddlePaddle/Paddle/pull/34979))
-   For the static graph, support partial ``program``, and generate the corresponding reverse ``program``.  ([#34395](https://github.com/PaddlePaddle/Paddle/pull/34395))
-   oneDNN function optimization
    - Add the support for oneDNN kernels with multiple operators, including ``clip``, ``slice``, ``split``, ``cast``, ``scale``, ``expand_v2``, ``sigmoid, matmul_v2``, ``PRelu`` forward and reverse oneDNN FP32, and oneNheN BF16. ([#35601](https://github.com/PaddlePaddle/Paddle/pull/35601), [#34332](https://github.com/PaddlePaddle/Paddle/pull/34332), [#34284](https://github.com/PaddlePaddle/Paddle/pull/34284), [#34216](https://github.com/PaddlePaddle/Paddle/pull/34216), [#34192](https://github.com/PaddlePaddle/Paddle/pull/34192),  [#33878](https://github.com/PaddlePaddle/Paddle/pull/33878), [#33584](https://github.com/PaddlePaddle/Paddle/pull/33584), [#33056](https://github.com/PaddlePaddle/Paddle/pull/33056), [#32975](https://github.com/PaddlePaddle/Paddle/pull/32975))
    - Add the implementation of Selected rows in SGD operator by using oneDNN AXPY. ([33632](https://github.com/PaddlePaddle/Paddle/pull/33632))
-   Support for ``bfloat16`` data type on the GPU with the Ampere architecture. ([#31232](https://github.com/PaddlePaddle/Paddle/pull/32132), [#32221](https://github.com/PaddlePaddle/Paddle/pull/32221), [#32542](https://github.com/PaddlePaddle/Paddle/pull/32542))
-   On the ``Conv`` operator, set the using of Tensor Core on the GPU with Ampere architecture. ([#34409](https://github.com/PaddlePaddle/Paddle/pull/34409))
-   Support ``paddle.device.cuda.current_stream().cuda_stream`` to get bare pointers.  ([#35813](https://github.com/PaddlePaddle/Paddle/pull/35813))
-   Add the ``paddle.optimizer.AdamW`` GPU fuse kernel, to support the layerwise learning rate function.  ([#35020](https://github.com/PaddlePaddle/Paddle/pull/35020), [#35569](https://github.com/PaddlePaddle/Paddle/pull/35569))
-   Support for using the Nvidia's cusparse library function in paddle. ([#35675](https://github.com/PaddlePaddle/Paddle/pull/35675))
-   Add ``paddle.full`` to support the ``int16`` type. ([#35619](https://github.com/PaddlePaddle/Paddle/pull/35619))
-   Optimize the GPU memory usage of ``paddle.nn.ClipGradByGlobalNorm``. ([#34586](https://github.com/PaddlePaddle/Paddle/pull/34586))
-   `reduce_sum` operator supports float16 type ([#32966](https://github.com/PaddlePaddle/Paddle/pull/32966))
-   `paddle.nn.CTCLoss`: Add two grad norm methods: `norm_by_total_logits_len` and `norm_by_batchsize`.  ([#34729](https://github.com/PaddlePaddle/Paddle/pull/34729/)) 
-   Add the public API recommended usages under each path. ([#33313](https://github.com/PaddlePaddle/Paddle/pull/33313), [#33308](https://github.com/PaddlePaddle/Paddle/pull/33308), [#32759](https://github.com/PaddlePaddle/Paddle/pull/32759), [#32695](https://github.com/PaddlePaddle/Paddle/pull/32695), [#32643](https://github.com/PaddlePaddle/Paddle/pull/32643), [#31912](https://github.com/PaddlePaddle/Paddle/pull/31912), [#32650](https://github.com/PaddlePaddle/Paddle/pull/32650), [#32034](https://github.com/PaddlePaddle/Paddle/pull/32034), [#33897](https://github.com/PaddlePaddle/Paddle/pull/33897)) 
-   Restore the original API accessibility under the `paddle.vision` path. ([#34432](https://github.com/PaddlePaddle/Paddle/pull/34432))
-   `paddle.vision.ops.deform_conv2d, paddle.vision.ops.DeformConv2D` : Add the support for the double input type.  ([#35330](https://github.com/PaddlePaddle/Paddle/pull/35330))
-   `paddle.fluid.contrib.layers.shuffle_batch` : Add the GPU Kernel implementation.  [#33938](https://github.com/PaddlePaddle/Paddle/pull/33938) 
-   For the existing APIs, add the public call paths `paddle.linalg.cholesky`, `paddle.linalg.norm`, and `paddle.linalg.inv`. ([#33420](https://github.com/PaddlePaddle/Paddle/pull/33420)) 
-   `paddle.reshape`: Support turning an empty `Tensor` shape into an empty `Tensor` of another shape. ([#36087](https://github.com/PaddlePaddle/Paddle/pull/36087))
-   `paddle.equal`: Add the support for `int`, `float`, and `bool` types for the second input. ([#35695](https://github.com/PaddlePaddle/Paddle/pull/35695))
-   ``paddle.io.DataLoader``: Add the support for persistent_worker mode. ([#34017](https://github.com/PaddlePaddle/Paddle/pull/34017))
-   Optimize ``l2_normalize``, ``p_norm``, ``elementwise_max``, ``prelu,clip_by_norm``, ``lars optimizer`` operators support the float16 computation.  ([#35576](https://github.com/PaddlePaddle/Paddle/pull/35576), [#35888](https://github.com/PaddlePaddle/Paddle/pull/35888), [#35888](https://github.com/PaddlePaddle/Paddle/pull/35888), [35532](https://github.com/PaddlePaddle/Paddle/pull/35532), [#35446](https://github.com/PaddlePaddle/Paddle/pull/35446), [#33280](https://github.com/PaddlePaddle/Paddle/pull/33280))
- Optimize the reading speed of flowers dataset from several minutes per batch to 1~3 seconds per batch.  ([#31408](https://github.com/PaddlePaddle/Paddle/pull/31408))
- Support the fuse allreduce sum function in `paddle.distributed.fleet.DistributedStrategy` when the `without_graph_optimize` switch is on.In the FP32, the performance increases by 3%. In the AMP, the performance increases by 8%. ([#34446](https://github.com/PaddlePaddle/Paddle/pull/34446)) 
- In `paddle.matmul`, switch underlying Op from matmul op to matmul_v2 op.  ([#36374](https://github.com/PaddlePaddle/Paddle/pull/36374))
- In `paddle.fft` module, add mkl_cdft and hipfft two computational backends.  ([#36537](https://github.com/PaddlePaddle/Paddle/pull/36537))
- Parameter `shifts` of `paddle.roll` supports `Tensor` as input.  ([#36537](https://github.com/PaddlePaddle/Paddle/pull/36537))
- `paddle.shape` supports plural type inputs. ([#36835](https://github.com/PaddlePaddle/Paddle/pull/36835))
- matmul_v2 supports quantization. ([#36469](https://github.com/PaddlePaddle/Paddle/pull/36469))
- Add `clip_op` support for `float16`. ([#36672](https://github.com/PaddlePaddle/Paddle/pull/36672))
- In `paddle.fft` module, add cache plan functionality to the cufft backend, optimizing performance. ([#36537](https://github.com/PaddlePaddle/Paddle/pull/36537))



#### IR(Intermediate Representation)

- Dynamic graph to static graph  
  - Optimize dynamic to static error reporting format, hide unnecessary error reporting stack at the framework level, add user code error line location identifier and context.  ([#35365](https://github.com/PaddlePaddle/Paddle/pull/35365), [#35320](https://github.com/PaddlePaddle/Paddle/pull/35320))
  - Optimize the conversion logic of the ``list.append`` syntax in the control flow. ([#35212](https://github.com/PaddlePaddle/Paddle/pull/35212)) 
  - Optimize the logic of dynamic to static training codes, upgrade the internal ``Program`` cache mechanism, and add an advance copy policy for input ``Tensor`` to improve training performance.   ([#34181](https://github.com/PaddlePaddle/Paddle/pull/34181), [#33796](https://github.com/PaddlePaddle/Paddle/pull/33796))
  - Optimize the internal actuator memory recycling strategy for dynamic to static graphs, reducing the GPU memory usage during training.  ([#34177](https://github.com/PaddlePaddle/Paddle/pull/34177))
  - Integrate the source codes of ``Gast`` triple dependency library, decoupling version dependencies.  ([#34556](https://github.com/PaddlePaddle/Paddle/pull/34556)) 
  - Display partial frame level error reporting information in case of dynamic-to-static error reporting. It is easier to locate the problem. ([#36765](https://github.com/PaddlePaddle/Paddle/pull/36765))
  - Remove duplicate temporary file removal function `remove_static_file()` in the dynamic to static error reporting module. ([#36375](https://github.com/PaddlePaddle/Paddle/pull/36375))
  - Optimize processing of `input_specs` parameter in RegisterPass, to support graph optimization as a matching subgraph condition.  ([#36453](https://github.com/PaddlePaddle/Paddle/pull/36453))

#### **Distributed training**

- Basic functions of distributed training
  - Enhance the check of the static graph pipeline parallel stage and persist var. ([#34193](https://github.com/PaddlePaddle/Paddle/pull/34193), [#34870](https://github.com/PaddlePaddle/Paddle/pull/34870), [#35453](https://github.com/PaddlePaddle/Paddle/pull/35453))
  - Optimize static graph pipeline parallel. In the 1F1B scheduling, the GPU memory does not increase as global batch size increases. ([#34230](https://github.com/PaddlePaddle/Paddle/pull/34230))
  - For the GPU Parameter Server, optimize the build phase hashmap. In the build phase, the performance increases by up to 7x on some tasks. ([#34175](https://github.com/PaddlePaddle/Paddle/pull/34175)) 
  - For the GPU Parameter Server, add the multi-stream parallel in the pull/push phase. ([#34276](https://github.com/PaddlePaddle/Paddle/pull/34276)) 
  - For the GPU Parameter Server, support the remote pull of parameters between machines in multi-machine training mode.  ([#35396](https://github.com/PaddlePaddle/Paddle/pull/35396))
  - For the CPU Parameter Server, support SSD storage. ([#33031](https://github.com/PaddlePaddle/Paddle/pull/33031))
  - `paddle.io.Dataset`: Support the dynamic library parsing data. ([#33969](https://github.com/PaddlePaddle/Paddle/pull/33969))
  - In the `paddle.distributed.fleet.dataset.DatasetBase`, add the consistency check function for generated data of the `use_var_list` and `pipe_command`.  ([#34463](https://github.com/PaddlePaddle/Paddle/pull/34463))
  - Add the consistency check between the `emd` dimension of `paddle.fluid.layers.embedding` and `emb` dimension of `sparse table` in `fleet`.  ([#34249](https://github.com/PaddlePaddle/Paddle/pull/34249))
  - Dynamic graph hybrid parallel supports for Pure FP16 training. ([#36707](https://github.com/PaddlePaddle/Paddle/pull/36707))
  - Static graph hybrid parallel supports dropout using a fixed random seed generator to ensure consistency of global variables and randomness of local variables in model parallel.  ([#36682](https://github.com/PaddlePaddle/Paddle/pull/36682))
  - Implement CPU parallelism and support for adding custom backend parameters when calling spawn or launch.  Available backend options are "gloo", "nccl", "bkcl", and "auto", for CPU parallel, GPU parallel, XPU parallel, and automatic selection by Paddle version, respectively.  ([#35745](https://github.com/PaddlePaddle/Paddle/pull/35745))
  - Optimize dynamic graph hybrid parallel HybridParallelClipGrad policy, to support 4D hybrid parallel + Pure FP16 training.  ([#36707](https://github.com/PaddlePaddle/Paddle/pull/36707))
  - Add SlotRecordDataset class to support GPU parameter server training.  ([#36710](https://github.com/PaddlePaddle/Paddle/pull/36710))
  - In the GPU parameter server building phase, support use of SlotRecordDataset. ([#36723](https://github.com/PaddlePaddle/Paddle/pull/36723))


- Static graph hybrid parallel
  - Optimize hybrid parallel loss scale and reduce the number of scale op insertions. ([#35775](https://github.com/PaddlePaddle/Paddle/pull/35775))
  - Optimize the pipeline scheduler, cache duplicate CPU jobs, and reduce CPU overhead.  ([#35680](https://github.com/PaddlePaddle/Paddle/pull/35680))
  - Optimize the number of times of checkpoint send/recv in pipeline parallel + recompute.  ([#34248](https://github.com/PaddlePaddle/Paddle/pull/34248))


#### **Others**

- Error debugging optimization
  - Unify the error reporting mechanism for third-party libraries, and optimize the error reporting messages for ``CURAND, CUDNN, CUBLAS, CUSOLVER, and NCCL``. This makes the error reporting more detailed and standardized. ([#33003](https://github.com/PaddlePaddle/Paddle/pull/33003), [#33743](https://github.com/PaddlePaddle/Paddle/pull/33743))
  - Optimize avx and no_avx related installation error messages to simplify redundant and complex contents. ([#33818](https://github.com/PaddlePaddle/Paddle/pull/33818)) 
  - Optimize the error report of the ``paddle.nn.functional.gather_tree``, ``paddle.nn.Transformer``, ``paddle.nn.TransformerDecoderLayer``, ``paddle.nn.TransformerEncoderLayer``, and ``paddle.nn.MultiHeadAttention``.  ([#34322](https://github.com/PaddlePaddle/Paddle/pull/34322), [#33859](https://github.com/PaddlePaddle/Paddle/pull/33859))
  - Support the configuration of ``FLAGS_check_nan_inf`` environment variable under dynamic graphs for runtime checking and localization of model ``nan`` and ``inf``.  ([#32635](https://github.com/PaddlePaddle/Paddle/pull/32635))
  - Remove the stack information introduced by Signal class error messages due to the capture of Signal, to avoid misleading users. ([#34842 ](https://github.com/PaddlePaddle/Paddle/pull/34842))
  - Fix error message for ``elementwise`` class operator when input x or y is an empty Tensor.  ([#33928](https://github.com/PaddlePaddle/Paddle/pull/33928))

- Model saving and loading
  - Fix the ``paddle.jit.save`` interface and model pruning logic. It is unnecessary to add an associated ``scale_op`` for output variables, and to properly export models containing outputs of type ``bool`` and ``float16``.  ([#35730](https://github.com/PaddlePaddle/Paddle/pull/35730), [#36132](https://github.com/PaddlePaddle/Paddle/pull/36132))
- Custom OP
  - Remove unnecessary ``cudaStreamSynchronize`` operations from ``paddle::Tensor's`` ``copy`` method, to improve performance.  ([#35802](https://github.com/PaddlePaddle/Paddle/pull/35802))
- Add C++ to support for GeneratePass development registration. The development mode is aligned with Python side. ([#36302](https://github.com/PaddlePaddle/Paddle/pull/36302))
- Automic SParsity

- Add `paddle.static.sparsity`, to support generating sparse parameters for `n:m` sparse mode. Currently, it only supports static graph ASP training. FP32 and FP16 on A100 are set with `1:2` and `2:4` sparse modes, respectively, to train saved sparse models, which can be used to accelerate inference tasks by calling TensorRT 8 based on the sparse Tensor Core of Ampere architecture. The current version provides a total of 5 APIs:  ([#32995](https://github.com/PaddlePaddle/Paddle/pull/32995)、[#33132](https://github.com/PaddlePaddle/Paddle/pull/33132)、[#33558](https://github.com/PaddlePaddle/Paddle/pull/33558)、[#36525](https://github.com/PaddlePaddle/Paddle/pull/36525))
  - `paddle.static.sparsity.calculate_density`: calculates the density of the input Tensor.  
  - `paddle.static.sparsity.decorate`: wraps the given optimizer as `OptimizerWithSparsityGuarantee`, automatically inserting necessary operations for the ASP workflow when calling `optimizer.minimize()`.    
  - `paddle.static.sparsity.prune_model`: prunes the parameters of the supported layers in `main_program` based on the mask generator function specified by `mask_algo`. 
  - `paddle.static.sparsity.set_excluded_layers`: sets the names of the parameters of layers that will not be trimmed.   
  - `paddle.static.sparsity.reset_excluded_layers`: resets the `excluded_layers` setting corresponding to `main_program`. 


### **(3) Performance optimization**

#### **Distributed training-static graph hybrid parallel**

- Optimize the AMP grey list when model parallel + AMP. Support the model parallel operator. The performance improves by 8%. ([#33660](https://github.com/PaddlePaddle/Paddle/pull/33660))
- Optimize the `device` property setting for reverse gradient accumulation in case of pipeline parallel. The performance improves by 1-3%. ([#33946](https://github.com/PaddlePaddle/Paddle/pull/33946))
- Optimize the debug part of the pipeline parallel executor. The performance improves by 60-140%.   ([#33948](https://gifothub.com/PaddlePaddle/Paddle/pull/33948))
- Support the `Program` cache in the pipeline parallel. The performance improves by 10-40%.  ([#33998](https://github.com/PaddlePaddle/Paddle/pull/33998), [#33954](https://github.com/PaddlePaddle/Paddle/pull/33954))
- Optimize the communication waiting for the pipeline parallel `send`. The performance improves by 0.3-2%.  ([#34086](https://github.com/PaddlePaddle/Paddle/pull/34086)) 
- Optimize the size of `send/recv` data volume in case of model parallel + pipeline parallel. The performance improves by 36% in the 8-machine test.  ([#34110](https://github.com/PaddlePaddle/Paddle/pull/34110))
- Optimize the cast of parameters in the hybrid parallel + AMP. It is controlled by `optimize_cast`. The performance improves by 5-7%.  ([#34965](https://github.com/PaddlePaddle/Paddle/pull/34965))
- Optimize the performance when pipeline parallel + recompute + amp. The performance improves by 13%.  ([#34519](https://github.com/PaddlePaddle/Paddle/pull/34519))
- Support the ``float16`` communication when pipeline parallel + data parallel. It is controlled by ``distributed_strategy.fp16_allreduce``. The performance improves by 13% performance improvement.  ([#34762](https://github.com/PaddlePaddle/Paddle/pull/34762))

#### **Operator optimization**

- Design and implement the generic Reduce CUDA algorithm. It is applied to 7 Reduce operators, increasing by 1.0x ~ 22.7x. ([#32697](https://github.com/PaddlePaddle/Paddle/pull/32697), [#32974](https://github.com/PaddlePaddle/Paddle/pull/32974), [#33267](https://github.com/PaddlePaddle/Paddle/pull/33267), [#32885](https://github.com/PaddlePaddle/Paddle/pull/32885), [#33144](https://github.com/PaddlePaddle/Paddle/pull/33144),  [#33761](https://github.com/PaddlePaddle/Paddle/pull/33761), [#33901](https://github.com/PaddlePaddle/Paddle/pull/33901), [#34143](https://github.com/PaddlePaddle/Paddle/pull/34143),  [#34436](https://github.com/PaddlePaddle/Paddle/pull/34436))
- Design and implement the generic Elementwise and Broadcast CUDA algorithms.  ([#32512](https://github.com/PaddlePaddle/Paddle/pull/32512), [#32928](https://github.com/PaddlePaddle/Paddle/pull/32928), [#33976](https://github.com/PaddlePaddle/Paddle/pull/33976), [#32148](https://github.com/PaddlePaddle/Paddle/pull/32148), [#32414](https://github.com/PaddlePaddle/Paddle/pull/32414)): Applied to 41 monadic and activation operators. ([#32348](https://github.com/PaddlePaddle/Paddle/pull/32348), [#32622](https://github.com/PaddlePaddle/Paddle/pull/32622), [#32823](https://github.com/PaddlePaddle/Paddle/pull/32823)). The performance improves by 1.1x ~ 1.4x. It is applied to 19 dualistic (9 basic computation class, 6 comparison class, and 4 logic class) operators. ([#33050](https://github.com/PaddlePaddle/Paddle/pull/33050), [33052](https://github.com/PaddlePaddle/Paddle/pull/33052), [#33053](https://github.com/PaddlePaddle/Paddle/pull/33053), [#33051](https://github.com/PaddlePaddle/Paddle/pull/33051), [#33089](https://github.com/PaddlePaddle/Paddle/pull/33089)) . The performance improves by 1.02x ~ 3.21x.  
- Optimize the ``roll`` operator CUDA implementation. The performance improves by more than 10% and 50% in case of single-dimensional and multi-dimensional inputs, respectively.  ([#32880](https://github.com/PaddlePaddle/Paddle/pull/32880))
- Optimize the ``roll`` operator index computation. The performance improves by 15% and 70% for single-dimensional and multi-dimensional input, respectively. ([#33909](https://github.com/PaddlePaddle/Paddle/pull/33909))
- Optimize the CUDA implementation of the `update_loss_scaling_op` operator. The performance improves by 2.06x.  ([#32554](https://github.com/PaddlePaddle/Paddle/pull/32554))
- Optimize the ``softmax_with_cross_entropy (hard label)`` GPU operator performance. The acceleration ratio is 1.0x ~ 10.0x.  ([#35660](https://github.com/PaddlePaddle/Paddle/pull/35660))
- Optimize the CPU implementation of ``index_select`` forward and inverse operators. The acceleration ratio is 2.09x ~ 9.34x. ([#32863](https://github.com/PaddlePaddle/Paddle/pull/32863),  [#32955](https://github.com/PaddlePaddle/Paddle/pull/32955))
- Optimize the CPU implementation of the ``batch_norm`` operator for 2-dimensional inputs. The acceleration ratio is 22.68x~30.00x.  ([#34585](https://github.com/PaddlePaddle/Paddle/pull/34585))
- Optimize the GPU performance of the ``batch_norm`` operator in the initialization method and 2-dimensional input. The acceleration ratio is 1.25x~25x.  ([#33851](https://github.com/PaddlePaddle/Paddle/pull/33851), [#33887](https://github.com/PaddlePaddle/Paddle/pull/33887))
- Optimize the ``log_softmax`` operator performance, and fix the related bug. The kernel performance improves by 4.22x~32.29x after optimization. ([#31630](https://github.com/PaddlePaddle/Paddle/pull/31630), [#32180](https://github.com/PaddlePaddle/Paddle/pull/32180), [#32396](https://github.com/PaddlePaddle/Paddle/pull/32396), [#32937](https://github.com/PaddlePaddle/Paddle/pull/32937))
- Optimize the ``concat_and_split`` operator, to solve the problem that computation and communication cannot overlap when training BERT on Hygon DCU chips in dynamic graphs. The performance of BERT distributed training on Hygon DCU chip increases by 27%. ([#33982](https://github.com/PaddlePaddle/Paddle/pull/33982))
- Optimize the ``fused_elemwise_act`` operator, with more than ten times performance improvement in the MB computing scale. ([#33480](https://github.com/PaddlePaddle/Paddle/pull/33480))

#### **Strategy optimization**

- Add the ``build_strategy.fix_op_run_order`` strategy, to immobilize the order of op execution. The speed of the ResNet model with single machine 8 cards increases by 1.8%. ([#34427](https://github.com/PaddlePaddle/Paddle/pull/34427))
- For the dynamic graph inverse computation, support and automatically enable partial operator inplace strategy. The performance of the dynamic graph gpt model pure float16 training increases by 4.8%. ([#35412](https://github.com/PaddlePaddle/Paddle/pull/35412))
- Optimize the dynamic graph performance by stripping logic executed only on static graphs from the execution path of dynamic graphs. ([#34024](https://github.com/PaddlePaddle/Paddle/pull/34024))
- For the IR Pass, optimize the capability exposed as a general-purpose capability. Support both single machine and distributed optimization.The performance improves by 3%-5% in GPT mixed parallel scenarios. ([#34955](https://github.com/PaddlePaddle/Paddle/pull/34955), [#35704](https://github.com/PaddlePaddle/Paddle/pull/35704), [#34730](https://github.com/PaddlePaddle/Paddle/pull/34730), [#34524](https://github.com/PaddlePaddle/Paddle/pull/34524))
- Optimize the ctc loss grad computation, increase the speed by ~3x. Correspondingly, the GPU memory usage increases.  ([#34729](https://github.com/PaddlePadle/Paddle/pull/34729))
- transformer encoder Performance Optimization
  - Optimization method: add `paddle.incubate.nn.FusedMultiHeadAttention` and `paddle.incubate.nn.FusedFeedForward`. In the implementation, q, k, v gemm fusion and multiple kernel fusion optimization techniques are used to improve performance of the transformer encoder.    
    - FusedAttention
      - Add `paddle.incubate.nn.functional.fused_multi_head_attention`, to support fusion computation of multi-head attention.  ([#35905](https://github.com/PaddlePaddle/Paddle/pull/35905) [35903](https://github.com/PaddlePaddle/Paddle/pull/35903) [#36803](https://github.com/PaddlePaddle/Paddle/pull/36803) [#36793](https://github.com/PaddlePaddle/Paddle/pull/36793) [36185](https://github.com/PaddlePaddle/Paddle/pull/36185))
      - Add `paddle.incubate.nn.FusedMultiHeadAttention` for layer networking of the fused multi-head attention.  ([#36498](https://github.com/PaddlePaddle/Paddle/pull/36498) )
      - This module uses q, k, v gemm fusion and bias add + dropout + residual add + layer_norm kernel fusion optimization techniques, resulting in 1.08x-1.45x acceleration. 

    - FusedFeedForward
      - Add `paddle.incubate.nn.functional.fused_feedforward`, to support feedforward fusion computation.  ([#36729](https://github.com/PaddlePaddle/Paddle/pull/36729) [#36730](https://github.com/PaddlePaddle/Paddle/pull/36730))
      - Add `paddle.incubate.nn.FusedFeedForward` for layer networking of fused feedforward.  ([#36776](https://github.com/PaddlePaddle/Paddle/pull/36776))
      - Performance is improved by about 1.04x~1.22x over pre-optimization.
      - Add `paddle.incubate.nn.FusedTransformerEncoderLayer`, to support layer networking by using fused multi-head attention and fused feedforward computation.  ([#36776](https://github.com/PaddlePaddle/Paddle/pull/36776))




### **(4) Troubleshooting**

#### API

-  Optimize the `depthwise_conv` numerical stability.  ([#35161](https://github.com/PaddlePaddle/Paddle/pull/35161))
-  Add the shape check at parameter creation, to ensure that the `size` of each axis of the parameter is greater than 0.  ([#33265](https://github.com/PaddlePaddle/Paddle/pull/33265))
-  Optimize the ``paddle.nn.LayerNorm`` computation, and fix the related data overflow bugs.  ([#34432](https://github.com/PaddlePaddle/Paddle/pull/34432), [#33658](https://github.com/PaddlePaddle/Paddle/pull/33658))
-  Support Windows application scenarios, integrate PaddlePaddle framework capabilities into MFC/QT/C# desktop software environments, and fix the bug in the process nesting that causes system crashes. ([#34312](https://github.com/PaddlePaddle/Paddle/pull/34312))
-  Fix the bug of the NLP model loss in the Reduce data initialization.  ([#34941](https://github.com/PaddlePaddle/Paddle/pull/34941))
-  Fix the bug when ``batch_size=1`` in ``paddle.nn.LayerNorm``. ([#35480](https://github.com/PaddlePaddle/Paddle/pull/35480))
-  Fix the bug of incorrectly catching an error when the input of ``paddle.static.nn.group_norm`` is empty. ([#35613](https://github.com/PaddlePaddle/Paddle/pull/35613))
-  Fix the bug of empty mean/variance when ``is_test=True`` in ``paddle.nn.functional.batch_norm``.  ([#35328](https://github.com/PaddlePaddle/Paddle/pull/35328))
-  Fix the out-of-bounds access bug when ``paddle.nn.functional.instance_norm`` and ``paddle.nn.functional.batch_norm`` are entered as null. ([#35341](https://github.com/PaddlePaddle/Paddle/pull/35341), [#34107](https://github.com/PaddlePaddle/Paddle/pull/34107))
-  Fix the bug where quantitative models do not count the output of ``paddle.nn.LayerNorm``.  ([#33610](https://github.com/PaddlePaddle/Paddle/pull/33610))
-  Fix the bug where ``paddle.nn.SyncBatchNorm.convert_sync_batchnorm()`` does not support 1D/3D.  ([#32989](https://github.com/PaddlePaddle/Paddle/pull/32989))
-  Fix the bug of failure to add the inverse in case of ``is_test=True`` in ``paddle.nn.BatchNorm1D, paddle.nn.BatchNorm2D, paddle.nn.BatchNorm3D``.  ([#32678](https://github.com/PaddlePaddle/Paddle/pull/32678))
-  Fix the bug where the `Tensor.cuda` does not support `device_id` configured to `None`.   ([#34416](https://github.com/PaddlePaddle/Paddle/pull/34416))
-  Fix the bug where the ``paddle.to_tensor`` does not support built-in types such as ``Tensor.dtype, core.Tensor``.  ([#31931](https://github.com/PaddlePaddle/Paddle/pull/31931), [#33430](https://github.com/PaddlePaddle/Paddle/pull/33430))
-  Fix the bug where the `paddle.nn.functional.log_softmax` does not support input dimension of 0.   ([#34635](https://github.com/PaddlePaddle/Paddle/pull/34635))
-  Fix the bug that the relative error between the CPU calculation result and accurate value of ``paddle.nn.GroupNorm`` under float32 is greater than that of 1e-5. ([#33176](https://github.com/PaddlePaddle/Paddle/pull/33176))
-  Fix the bug where the returned result is not 0 when the parameter ``offset`` exceeds the dimension size in the ``paddle.trace``, and fix the bug of the stack overflow when the parameters ``axis1`` and ``axis2`` entered are illegal values. ([#33922](https://github.com/PaddlePaddle/Paddle/pull/33922), [#35419](https://github.com/PaddlePaddle/Paddle/pull/35419))
-  Fix the bug where the output type is not int when the ``paddle.sum`` input parameter is the bool type.The output type is wrong when the input parameter type and output parameter type are inconsistent and the number of reduce elements corresponding to the axis is 1. ([#34313](https://github.com/PaddlePaddle/Paddle/pull/34313), [#36123](https://github.com/PaddlePaddle/Paddle/pull/36123))
-  Fix the bug of the division by 0 error and array out-of-bound when ``paddle.nn.conv2d/conv3d/conv2d_transpose/conv3d_transpose`` is the illegal input. ([#35337](https://github.com/PaddlePaddle/Paddle/pull/35337))
-  Fix the heap buffer overflow bug on illegal input of ``paddle.nn.conv2d_transpose``.  ([#35340](https://github.com/PaddlePaddle/Paddle/pull/35340))
-  Fix the bug where writing a null address to ``paddle.bmm`` causes the program to crash at runtime.  ([#35098](https://github.com/PaddlePaddle/Paddle/pull/35098))
-  Fix the bug when the ``cast`` operator cannot support Tensor conversion from int16 to float32.  ([#35156](https://github.com/PaddlePaddle/Paddle/pull/35156))
-  Fix the bug where the` assign` does not support float16 or uint8. ([#35153](https://github.com/PaddlePaddle/Paddle/pull/35153))
-  Fix the bug of `concat`'s tendency to overflow when the input is greater than shape tensor.  ([#34319](https://github.com/PaddlePaddle/Paddle/pull/34319))
-  Fix the bug where the `concat` in dynamic graphs does not support empty tensor as an input.  ([#35845](https://github.com/PaddlePaddle/Paddle/pull/35845))
-  Fix the bug where the ``paddle.where`` does not support broadcast.  ([#35092](https://github.com/PaddlePaddle/Paddle/pull/35092))
-  Fix the bug of ``paddle.reshape`` not checking input legality in the empty tensor. ([#35642](https://github.com/PaddlePaddle/Paddle/pull/35642))
-  Fix the bug of ``layernorm`` operator mis-matching with cuda kernel in the large shape.  ( [#33748](https://github.com/PaddlePaddle/Paddle/pull/33748))
-  Fix the bug of wrong setting of stop_gradient property in the static graph of ``random`` class operator. ( [#33959](https://github.com/PaddlePaddle/Paddle/pull/33959))
-  Fix the bug of wrong behavior of ``split`` operator with empty tensor input. ([#334356](https://github.com/PaddlePaddle/Paddle/pull/334356))
-  Fix the GPU memory leak bug in tensor's slice left-value assignment. ([#35013](https://github.com/PaddlePaddle/Paddle/pull/35013))
-  Fix the bug of the dynamic graph Layers not being used bycloudpickle dump and load. ([#35538](https://github.com/PaddlePaddle/Paddle/pull/35538))
-  Fix the bug of division by zero error in the illegal parameter settings for simple_rnn_cell, gru_cell, and lstm_cell APIs. ([#34627](https://github.com/PaddlePaddle/Paddle/pull/34627))
-  Fix the bug of the null pointer dereference in case of illegal input of ``paddle.nn.functional.linear``.  ([#34696](https://github.com/PaddlePaddle/Paddle/pull/34696))
-  Fix the memory out-of-bounds bug of the ``paddle.strided_slice``, ``paddle.transpose``. ([#35062](https://github.com/PaddlePaddle/Paddle/pull/35062), [#35079](https://github.com/PaddlePaddle/Paddle/pull/35079))
-  Fix the bug of the division by 0 error when the ``roll`` operator has an illegal input. ([#34499](https://github.com/PaddlePaddle/Paddle/pull/34499))
-  Fix an array out-of-bounds bug in the illegal input of the ``gather`` operator. ([#34096](https://github.com/PaddlePaddle/Paddle/pull/34096), [#34138](https://github.com/PaddlePaddle/Paddle/pull/34138), [#34200](https://github.com/PaddlePaddle/Paddle/pull/34200))
-  Fix the bug of division by 0 error in the illegal input of the ``prelu``, ``softlax`` operators. ([#34499](https://github.com/PaddlePaddle/Paddle/pull/34499))
-  Fix the bug where the ``split`` operator does not perform a legality check on input parameters.  ([#34630](https://github.com/PaddlePaddle/Paddle/pull/34630))
-  Fix the bug where the ``memcpy`` operator does not support Hygon DCU chips.  ([#35394](https://github.com/PaddlePaddle/Paddle/pull/35394))
-  Fix the bug of training error reporting of the ``slice`` operator when ``batch_size=1``. ([#34265](https://github.com/PaddlePaddle/Paddle/pull/34265))
-  Fix the overflow bug of the ``reduce_sum`` operator in the AMP.  ([#33960](https://github.com/PaddlePaddle/Paddle/pull/33960))
-  Fix the ANSI escape code error on windows.  ([#33689](https://github.com/PaddlePaddle/Paddle/pull/33689))
-  Fix the inconsistency bug between ``paddle.hub`` parsed file names and downloaded and saved files.  ([#33214](https://github.com/PaddlePaddle/Paddle/pull/33214))
-  Fix the memory leak bug when inputting empty tensor for ``matmul``, ``diag_embed``, and ``auc`` operators.  ([#34978](https://github.com/PaddlePaddle/Paddle/pull/34978))
-  Fix the bug of large computational accuracy error of broadcast for ``paddle.less_equal, paddle.less_than, paddle.greater_equal, and paddle.greater_than``. ([#32941](https://github.com/PaddlePaddle/Paddle/pull/32941))
-  Fix the crash bug of ``interpolate`` operator in case of a large input shape. ([#35577](https://github.com/PaddlePaddle/Paddle/pull/35577))
-  Fix legality check for ``interpolate``, ``unfold``, and ``spectral_norm`` operators in case of empty tensor input.  ([#33941](https://github.com/PaddlePaddle/Paddle/pull/33941), [#34943](https://github.com/PaddlePaddle/Paddle/pull/34943), [#35005](https://github.com/PaddlePaddle/Paddle/pull/35005))
-  Fix a possible negative sign (integer overflow) in `paddle.flops` when computing the output FLOPs. ([#33576](https://github.com/PaddlePaddle/Paddle/pull/33576))
-  Fix the bug of reporting an error when ``paddle.summary`` encounters a layer whose return value contains a non-Tensor element. ([#34160](https://github.com/PaddlePaddle/Paddle/pull/34160))
-  Fix the bug where the output shape is calculated incorrectly when the ``pool`` operator is entered illegally.  ([#35106](https://github.com/PaddlePaddle/Paddle/pull/35106))
-  Fix the legality check bug of the input shape for ``unfold, dice_loss, and reshape`` operators.  ([#34673](https://github.com/PaddlePaddle/Paddle/pull/34673), [#34757](https://github.com/PaddlePaddle/Paddle/pull/34757), [#35016](https://github.com/PaddlePaddle/Paddle/pull/35016))
-  Fix the input zero tensor bug of the ``unique, and unstack`` operators. ([#36021](https://github.com/PaddlePaddle/Paddle/pull/36021))
-  Fix the bug when the reverse input of stack operator is null. ([#362877](https://github.com/PaddlePaddle/Paddle/pull/32877))
-  Fix the bug of the division by 0 error in the CPU execution when the shape of the input Tensor of ``paddle.inverse`` is ``[0, 0, 0]``.  ([#34996](https://github.com/PaddlePaddle/Paddle/pull/34996))
-  Fix the bug of the CUDA error reported by ``paddle.nn.functional.grid_sample`` for special input cases. ([#33100](https://github.com/PaddlePaddle/Paddle/pull/33100))
-  Fix a compile-time dimension calculation error in ``paddle.flatten`` for special input cases of static graphs. ([#35321](https://github.com/PaddlePaddle/Paddle/pull/35321))
-  Fix a compile-time check error in ``paddle.nn.conv2d/conv3d/conv2d\_transpose/conv3d\_transpose`` when calculating output shape.  ([#35693](https://github.com/PaddlePaddle/Paddle/pull/35693))
-  Fix the bug where ``paddle.data.flowers`` is prone to data reading errors in multi-card training situations.  ([#33738](https://github.com/PaddlePaddle/Paddle/pull/33738))
-  Fix the bug that the loss is nan when the pact quantizes the se module. ([#35392](https://github.com/PaddlePaddle/Paddle/pull/35392))
-  Fix the bug of error reporting in the quantization `flatten_contiguous_range`. ([35410](https://github.com/PaddlePaddle/Paddle/pull/35410))
-  Fix the bug of pact quantization in dynamic graph mode.  ([#35407](https://github.com/PaddlePaddle/Paddle/pull/35407))
-  Fix the bug of the error report by channel-wise quantization bert. ([#34948](https://github.com/PaddlePaddle/Paddle/pull/34948))
-  Fix the bug with quantization when all parameters are 0. ([#34647](https://github.com/PaddlePaddle/Paddle/pull/34647))
-  Fix a bug in channel-wise quantization when the number of channels is 1. ([#33753](https://github.com/PaddlePaddle/Paddle/pull/33753))
-  Fix the bug of thread insecurity of the dynamic graph ``@no_grad``.  ([#34649](https://github.com/PaddlePaddle/Paddle/pull/34649))
-  Fix the bug where the ``paddle.grad`` interface will hang in some scenarios. ([#34023](https://github.com/PaddlePaddle/Paddle/pull/34023))
-  Fix the bug of shape derivation in ``paddle.masked_select`` in static graphs. ([#33167](https://github.com/PaddlePaddle/Paddle/pull/33167))
-  Fix the bug of ``paddle.slice`` not supporting ``numpy.ndarray`` type index in some scenarios, and error when ``axes`` is the ``tuple`` type. ([#35748](https://github.com/PaddlePaddle/Paddle/pull/35748), [#35267](https://github.com/PaddlePaddle/Paddle/pull/35267))
-  Fix the `set_value` reverse gradient truncation bug. ([#34304](https://github.com/PaddlePaddle/Paddle/pull/34304))
-  Fix the ``paddle.regularizer.L1Decay`` duplicate gradient setting bug in the non-inplace computation.  ([32710](https://github.com/PaddlePaddle/Paddle/pull/32710))
-  Fix the bug with learning rate not taking effect when grouping ``adamw`` parameters. ([#34468](https://github.com/PaddlePaddle/Paddle/pull/34468))
-  Optimize illegal ``dilate`` input check in convolution class APIs. ([#35894](https://github.com/PaddlePaddle/Paddle/pull/35894))
-  Fix the bug of the `paddle.io.DataLoader` iteration mid-break error reporting. ([#34501](https://github.com/PaddlePaddle/Paddle/pull/34501)) DataLoader memory leak bug. ([#34140](https://github.com/PaddlePaddle/Paddle/pull/34140)) DataLoader wrongly reporting the warning information. ([#33712](https://github.com/PaddlePaddle/Paddle/pull/33712))          DataLoader sub-process random state consistency bug. ([#33310](https://github.com/PaddlePaddle/Paddle/pull/33310))
-  Fix drop_last not taking effect in IterableDataset. ([#34801](https://github.com/PaddlePaddle/Paddle/pull/34801))
-  Fix the bug with optimizer state recovery caused by ``paddle.optimizer.lr.LRScheduler``.   ( [#33984](https://github.com/PaddlePaddle/Paddle/pull/33984))
-  Fix the bug of using ``axis`` for infershape in ``gather`` operator. ([#33413](https://github.com/PaddlePaddle/Paddle/pull/33413))
-  Fix a bug of getting stuck in Executor where fetch_list type is a tuple. ([#35726](https://github.com/PaddlePaddle/Paddle/pull/35726))
-  Fix the ``paddle.nn.GroupNorm`` divided by zero error, and add channel with the exact division check by group. ([#35644](https://github.com/PaddlePaddle/Paddle/pull/35644))
-  Fix the bug with referencing the freed memory in tensor formatter. ([#35399](https://github.com/PaddlePddle/Paddle/pull/35399))
-  Fix the bug of the ``beta`` parameter precision loss at ``float64`` precision for the Adam optimizer.  ([#33381](https://github.com/PaddlePaddle/Paddle/pull/33381))
-  Fix the precision misalignment bug caused by unbroadcasted initialization of tensor parallel non-tangent parameters. ([#35326](https://github.com/PaddlePaddle/Paddle/pull/35326))
-  Migrate the ``topk`` operator in the ``paddle.static.accuracy`` API to the ``topk_v2`` operator.  ([#35494](https://github.com/PaddlePaddle/Paddle/pull/35494))
-  Migrate the ``expand`` operator to ``tile`` operator in ``paddle.nn.dynamic_decode``, and ``topk`` operator to ``topk_v2`` operator in the ``paddle.nn.BeamSearchDecoder``. ([#35656](https://github.com/PaddlePaddle/Paddle/pull/35656))
-  Migrate the one_hot operator in ``paddle.nn.functional.dice_loss`` API to the ``one_hot_v2`` operator. ([#35734](https://github.com/PaddlePaddle/Paddle/pull/35734))
-  Fix the bug of usage in the static graph mode in ``paddle.summary``. ([#35303](https://github.com/PaddlePaddle/Paddle/pull/35303))
-  Fix the multi-card startup bug in ``paddle.Model.prepare`` static graph mode.  ([#34311](https://github.com/PaddlePaddle/Paddle/pull/34311))
- Fix error report of `paddle.nn.functional.cross_entropy` when `weight` is given and `axis` is specified as a legal dimension other than -1. ([#36647](https://github.com/PaddlePaddle/Paddle/pull/36647))
- Fix a bug with `paddle.utils.dlpack.to_dlpack` that prevents it from encoding multidimensional `Tensor`, and fix a bug with its generated DLPack objects not being shared across deep learning frameworks. ([#36177](https://github.com/PaddlePaddle/Paddle/pull/36177))
- Fix a bug in the `sample` method using `paddle.distribution.Categorical`, specifically, due to an out-of-bounds array access in the multinomial op's cuda kernel. The bug causes access to values beyond the subscript of the array, causing an error to be reported. ([#36511](https://github.com/PaddlePaddle/Paddle/pull/36511))
- Fix a bug in the dynamic graph `_BatchNormBase` base class where the default_dtype is modified, resulting in the wrong type of subsequent networking parameters. Affected APIs are `paddle.nn.BatchNorm1D`, `paddle.nn.BatchNorm2D`, ` paddle.nn.BatchNorm3D`, and `paddle.nn.SyncBatchNorm`. The specific reason is that when `get_default_dtype() == 'float16'`, the default parameter data type is modified by `set_default_dtype('float32')`. The parameter type of dynamic graph networking is created by default_dtype. Therefore, when the default parameter type is modified, subsequent networking parameter type is consequently incorrect.  ([#36376](https://github.com/PaddlePaddle/Paddle/pull/36376))
- Fix an exception in `paddle.nn.functional.grid_sample` caused by special input. ([#36625](https://github.com/PaddlePaddle/Paddle/pull/36625))
- Fix calculation error of `paddle.fft.ffft`, `paddle.fft.ifft`, `paddle.fft.rfft` , `paddle.fft.irfft`, `paddle.fft.hfft`, and `paddle.fft.ihfft` when input ` axis=0`. ([#36537](https://github.com/PaddlePaddle/Paddle/pull/36537))
- Fix a bug of errors of `paddle.fft.fftshift` and `paddle.fft.ifftshift` under static graphs.  ([#36537](https://github.com/PaddlePaddle/Paddle/pull/36537))
- Fix a bug where `paddle.fft.ifftshift` is not calculated correctly. ([#36835](https://github.com/PaddlePaddle/Paddle/pull/36835))
- Fix error message prompt for `paddle.nn.functional.pad` in `replicate` mode. ([#36531](https://github.com/PaddlePaddle/Paddle/pull/36531))



#### IR(Intermediate Representation)

- Dynamic graph to static graph
  - Fix an abnormal growth of GPU memory under ``paddle.no_grad`` semantics after dynamic to static. ([#35725](https://github.com/PaddlePaddle/Paddle/pull/35725))
  - Fix a misidentification and conversion bug in the ``paddle.no_grad`` interface.  ([#34136](https://github.com/PaddlePaddle/Paddle/pull/34136)) 
  - Fix a bug of reporting an error in dynamic to static training when stop_gradient=True is set in the middle of the model in some scenarios. ([#36353](https://github.com/PaddlePaddle/Paddle/pull/36353))
  - Fix a bug of reporting an error when checking the return result in some scenarios where the control flow “if” is converted. ([#36830](https://github.com/PaddlePaddle/Paddle/pull/36830))
  - Fix a bug that the return type changes unexpectedly due to additional dynamic to static aligning in the return length when “ifelse” branch returns unequal results. ([#36565](https://github.com/PaddlePaddle/Paddle/pull/36565))
  - Fix a bug where video memory will keep growing in train mode and no_grad contexts after loading a model via the jit.save/load interface. ([#36463](https://github.com/PaddlePaddle/Paddle/pull/36463))

#### **Distributed training**

- Basic functions of distributed training
  - Fix a potential stack overflow bug in the graph engine. ([#33055](https://github.com/PaddlePaddle/Paddle/pull/33055)) 
  - Fix a potential deadlock bug in the distributed training. ([#34461](https://github.com/PaddlePaddle/Paddle/pull/34461))
  - Fix the bug where tensor parallel is incorrectly sliced in the multi-headed attention computation of transformer class models. Optimize the speed of tensor parallel in mixed precision computations. ([#33015](https://github.com/PaddlePaddle/Paddle/pull/33015)) 
  - Fix the bug where the norm of non-distributed vars is computed for multiple times when using `paddle.nn.ClipGradientByGlobalNorm` in the model parallel. ([#35713](https://github.com/PaddlePaddle/Paddle/pull/35713))
  - Fix the bias addition position error in the row slice in the model parallel `paddle.distributed.split` Parallel Linear. ([#35186](https://github.com/PaddlePaddle/Paddle/pull/35186))
  - Fix the possible hang bug in the pipeline parallel initialization communication group. ([#33476](https://github.com/PaddlePaddle/Paddle/pull/33476))
  - Fix the bug where the `Tensor` GPU memory in pipeline parallel is released before it is actually used. ([#33996](https://github.com/PaddlePaddle/Paddle/pull/33996))
  - Fix the bug where the pipeline parallel reverse gradient accumulation `op_device` is empty.  ([#33875](https://github.com/PaddlePaddle/Paddle/pull/33875))
  - Fix the bug with pipeline parallel running `sub-block` errors.  ([#32727](https://github.com/PaddlePaddle/Paddle/pull/32727))
  - Fix the bug where the pipeline parallel reverse gradient accumulation `op_device` is empty. ([#33875](https://github.com/PaddlePaddle/Paddle/pull/33875))
  - Fix an occasional hang bug when initializing Sharding parallel communication. ([#33327](https://github.com/PaddlePaddle/Paddle/pull/33327))
  - Fix the `paddle.distributed.barrier` synchronization flow error bug.  ([#33476](https://github.com/PaddlePaddle/Paddle/pull/33476))
  - Fix the `paddle.distributed.alltoall` communication group setting error bug. ([#32890](https://github.com/PaddlePaddle/Paddle/pull/3492890))
  - Fix a precision misalignment caused by a static graph tensor parallel parameter initial swap broadcast error. ([35326](https://github.com/PaddlePaddle/Paddle/pull/35326))
  - Fix the bug where dynamic graph data parallel does not support custom operators such as `recompute` inheriting from `PyLayer` class. ([#35401](https://github.com/PaddlePaddle/Paddle/pull/35401))
  - Fix the hang bug in case of pipeline parallel + data parallel in the mixed parallel. ([#34142](https://github.com/PaddlePaddle/Paddle/pull/34142))
  - Fix the `fleet.get_loss_scaling` failure bug in case of enabling AMP.  ([#33935](https://github.com/PaddlePaddle/Paddle/pull/33935))
  - Fix the Connection Refused problem caused by a `fleet` multi-machine master not waiting for other nodes to be ready. ([#32889](https://github.com/PaddlePaddle/Paddle/pull/32889))
  - Fix the bug where the distributed prediction `infer_from_dataset` still updates parameter gradients. ([#35698](https://github.com/PaddlePaddle/Paddle/pull/35698))
  - Fix the bug in `data_feed` where the dense feature LOD attribute is incorrectly set. ([#35000](https://github.com/PaddlePaddle/Paddle/pull/35000))
  - Fix the save bug with the `gradient_merge_cond` variable when using `gradientmerge` for static graphs. ([#35578](https://github.com/PaddlePaddle/Paddle/pull/35578))
  - Fix the save bug with the `paddle.hub` download file name and the` nt_merge_cond variable`.  ([#35578](https://github.com/PaddlePaddle/Paddle/pull/35578))
  - Fix the bug of unclearly reporting an error when `fleet` is enabled with `dump_slot`. ([#34173](https://github.com/PaddlePaddle/Paddle/pull/34173))
  - Fix the RCCL bug on Hygon DCU chips in the hybrid parallel training. ([#32808](https://github.com/PaddlePaddle/Paddle/pull/32808))
  - Fix GPU parameter server exit error reporting bug. ([#33724](https://github.com/PaddlePaddle/Paddle/pull/33724))
  - Fix the bug of unavailability of upload/download function of the hdfs tool. ([#33903](https://github.com/PaddlePaddle/Paddle/pull/33903))
  - Fix the bug of the GPU parameter server getting stuck during training because the sample cannot exactly divide the worker number. ([#32640](https://github.com/PaddlePaddle/Paddle/pull/32640))
  - Fix the GPU parameter server error reported by using non-0 card training. ([#33078](https://github.com/PaddlePaddle/Paddle/pull/33078))
  - Fix the bug of the delta score and scale show in the GPU Parameter Server. ([#33492](https://github.com/PaddlePaddle/Paddle/pull/33078), [#33492](https://github.com/PaddlePaddle/Paddle/pull/33492))
  - Fix the bug with GPU Parameter Server not merging dense after training, in incorrect g2sum calculation. For data norm, add the optimize op. ([#35029](https://github.com/PaddlePaddle/Paddle/pull/35029))
  - Fix an error reported if the gradient is empty when using the fuse all reduce ops switch.  ([#36231](https://github.com/PaddlePaddle/Paddle/pull/36231))
  - Fix a bug with dist_transformer files showing undefined variables. ([#36211](https://github.com/PaddlePaddle/Paddle/pull/36211))

- Dynamic graph hybrid parallel
  - Fix the precision error in pipeline parallel due to communication asynchronization. [#35556](https://github.com/PaddlePaddle/Paddle/pull/35556)
  - Fix the precision exception bug in ``paddle.distributed.fleet.meta_parallel.RowParallelLinear`` reverse computation under tensor parallel. [#33207](https://github.com/PaddlePaddle/Paddle/pull/33207)
  - Fix a bug in tensor parallel causing parameter initialization error and precision exception due to randomness control error.  [#32897](https://github.com/PaddlePaddle/Paddle/pull/32897) ([#32897](https://github.com/PaddlePaddle/Paddle/pull/32897))
  - Fix the random hang bug when creating a communication group with ``paddle.distributed.new_group()``.  [#33141](https://github.com/PaddlePaddle/Paddle/pull/33141)
  - Fix the bug of causing an error in traversing the reverse graph to resolve control flow networking under data parallel.  [#32715](https://github.com/PaddlePaddle/Paddle/pull/32715)
  - Fix the bug of causing an error when synchronizing the parameters of each process under data parallel. [#33955](https://github.com/PaddlePaddle/Paddle/pull/33955)

- Static graph hybrid parallel
  - Fix a slice error in TensorParallel in Multi-Head Attention networks, and optimize the training speed when TensorParallel is used together with mixed precision. ([#32897](https://github.com/PaddlePaddle/Paddle/pull/32897))

#### **Others**

- Custom OP
  - Fix the bug where the ``cast`` method of ``paddle::Tensor`` does not take effect in the GPU.  ([#34884](https://github.com/PaddlePaddle/Paddle/pull/34884))
  - Fix the bug where custom operators cannot load multiple modules at the same time.  ([#34505](https://github.com/PaddlePaddle/Paddle/pull/34505))
  - Fix the bug where the ``PADDLE_WITH_CUDA`` macro does not take effect in co-compiling of .cc and .cu files. ([#35448](https://github.com/PaddlePaddle/Paddle/pull/35448))
- Remove changes to ``logging`` library global settings.  ([#32673](https://github.com/PaddlePaddle/Paddle/pull/32673))
- Add ``GlooParallelContext``, to adapt the ``Reducer`` module logic, and provide underlying communication component support for ``DataParallel`` subsequently supporting CPU parallel later.  ([#35154](https://github.com/PaddlePaddle/Paddle/pull/35154))
- Migrate `top_k` op in `paddle.metric.accuracy` to `top_k_v2` op.   ([#35789](https://github.com/PaddlePaddle/Paddle/pull/35789))
- Fix the bug where the default `attr` is not found running under `MKLDNN`. ([#34567](https://github.com/PaddlePaddle/Paddle/pull/34567))
- Fix the bug in `optimizer` where `device_key` is not added to the `clear_float_status` OP. ([#34431](https://github.com/PaddlePaddle/Paddle/pull/34431))



## **4. Deployment Direction (Paddle Inference)**

### **(1) New features**

#### **Back-end capability enhancement**

- Add the dynamic shape auto-configuration function in TensorRT sub-graph mode. Add TensorRT offline tune dynamic shape setting method. For scenarios where the model is cut into multiple TensorRT sub-graphs, improve ease of use. [#34806](https://github.com/PaddlePaddle/Paddle/pull/34806) [#35771](https://github.com/PaddlePaddle/Paddle/pull/35771), For examples, see the [demo](https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/c%2B%2B/paddle-trt/tuned_dynamic_shape).

  - The basic idea of the ease of use optimization: to use Paddle to run natively to statistically calculate the shape ranges of all temporary tensors in the graph for the batch data input by the user, and set the statistically calculated shape ranges to the input of TensorRT sub-graphs, thus avoiding the user to manually calculate the shape ranges of the input tensors of internal sub-graphs and improving ease of use.
    - Basic process of offline tuned dynamic shape: After the user code is completed, set the config, enable the shape range collection capability c++ interface `config. CollectShapeRangeInfo("shape_range.pbtxt")` or python interface `config. collect_shape_range_info('shape_range.pbtxt')`, to store the obtained shape range locally in prototxt format, modify the config to disable shape collection, and enable tensorrt and dynamic shape capability, c++ interface `config. EnableTunedTensorRtDynamicShape("shape_range.pbtxt", true)` or python interface `config.enable_tuned_tensorrt_dynamic_shape('shape_range.pbtxt', True)`. Thus, run run directly.


- Add native support for Ascend series hardware
  - sub-graphs are accessed to ascend310 hardware [#35226](https://github.com/PaddlePaddle/Paddle/pull/35226) by supporting Paddle-Lite NNAdapter. For the example, see the [demo](https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/c%2B%2B/ascend310_lite_subgraph/image_classification_demo).
  - New Ascend 910 inference support [#34101](https://github.com/PaddlePaddle/Paddle/pull/34101)
- Add pool3d OP to support for TensorRT. ([#36545](https://github.com/PaddlePaddle/Paddle/pull/36545))

### **(2) Function optimization**

#### **Framework and API updates**

- Quantification support
  - Refactor dynamic graph quantization inference pass, to support non-analog quantization OP and analog quantization OP. ([#35907](https://github.com/PaddlePaddle/Paddle/pull/35907))
  - Add int8 for analog quantized OP matmul (the case where weights are multiplied by tensor).  ([#34359](https://github.com/PaddlePaddle/Paddle/pull/34359))
  - Fix a bug that MobileNetV3 model "Loss” out of NAN during quantization training due to the quantization parameter being 0.  ([#36763](https://github.com/PaddlePaddle/Paddle/pull/36763))

- API enhancements
  - Refactor GO API based on new version of CAPI, [#33113](https://github.com/PaddlePaddle/Paddle/pull/33113). For the example, see the [demo](https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/go/resnet50).
  - Predict python api `copy_from_cpu` and `copy_to_cpu` interfaces to support float16 data types . ([#34676](https://github.com/PaddlePaddle/Paddle/pull/34676))
  - Add `config.Summary()` interface to print config configuration information. ([#34122](https://github.com/PaddlePaddle/Paddle/pull/34122))
  - In the prediction library `version.txt`, record trt version information patch, e.g., v7.2.3.4 instead of v7. ( [#33690](https://github.com/PaddlePaddle/Paddle/pull/33690))

- Library volume compression
  - In the Linux, the volume of the prediction library is pruned by strip, and the volume is compressed by 30m. ([#34895](https://github.com/PaddlePaddle/Paddle/pull/34895))

- Other updates
  - Add the helper tool to catch runtime exceptions and convert them to the appropriate error state.  ([#35624](https://github.com/PaddlePaddle/Paddle/pull/35624))
  - Add the related base data structure to enhance the accuracy of the PaddlePaddle operator definition. ([#33098](https://github.com/PaddlePaddle/Paddle/pull/33098))

#### **Back-end capability enhancement**

- CPU related updates
  - Upgrade oneDNN version to 2.3.2. ( [#35040](https://github.com/PaddlePaddle/Paddle/pull/35040))
  - Add the support of quant-aware LSTM oneDNN INT8 models. ([#35382](https://github.com/PaddlePaddle/Paddle/pull/35382))
  - Add the support of post-training LSTM oneDNN INT8 models. ([#35334](https://github.com/PaddlePaddle/Paddle/pull/35334), [#33295](https://github.com/PaddlePaddle/Paddle/pull/33295))
  - Add the support of fusion_gru and multi_gru fusion and post-training INT8. ([#33749](https://github.com/PaddlePaddle/Paddle/pull/33749))
  - Optimize the cache mechanism of oneDNN. ([#35664](https://github.com/PaddlePaddle/Paddle/pull/35664),  [#35331](https://github.com/PaddlePaddle/Paddle/pull/35331), [#35132](https://github.com/PaddlePaddle/Paddle/pull/35132), [#35030](https://github.com/PaddlePaddle/Paddle/pull/35030), [#35002](https://github.com/PaddlePaddle/Paddle/pull/35002), [#34830](https://github.com/PaddlePaddle/Paddle/pull/34830), [#33515](https://github.com/PaddlePaddle/Paddle/pull/33515), [#33048](https://github.com/PaddlePaddle/Paddle/pull/33048), [#32922](https://github.com/PaddlePaddle/Paddle/pull/32922), [#32499](https://github.com/PaddlePaddle/Paddle/pull/32499))
  - This is implemented by adding multiple op (e.g., clip, scale, etc.) of oneDNN kernel. In the ch_ppocr_mobile_v1.1_det_infer, DPN68, fastscnn, hrnet, HRNet_W18_C, icnet, Res2Net50_26w_4s, and ssdlite_mobilenet_v3_large models, the single core performance of Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz increases by 47.8% in the oneDNN enabling against disabling. ([#35601](https://github.com/PaddlePaddle/Paddle/pull/35601), [#32975](https://github.com/PaddlePaddle/Paddle/pull/32975))
  - Optimized oneDNN LSTM INT8 model with 1.59x performance improvement on Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz single core than that of the FP32 LSTM model. ([#35382](https://github.com/PaddlePaddle/Paddle/pull/35382), [#35334](https://github.com/PaddlePaddle/Paddle/pull/35334), [#34820](https://github.com/PaddlePaddle/Paddle/pull/34820), [#34137](https://github.com/PaddlePaddle/Paddle/pull/34137))


- GPU and TensorRT sub-graph engine related updates

  - Added support for TensorRT 8.0. We will drop support for TensorRT 6.x in a future release.  ([#34403](https://github.com/PaddlePaddle/Paddle/pull/34403), [#34294](https://github.com/PaddlePaddle/Paddle/pull/34294), [#34157](https://github.com/PaddlePaddle/Paddle/pull/34157), [#33777](https://github.com/PaddlePaddle/Paddle/pull/33777), [#33680](https://github.com/PaddlePaddle/Paddle/pull/33680), [#33662](https://github.com/PaddlePaddle/Paddle/pull/33662), [#33654](https://github.com/PaddlePaddle/Paddle/pull/33654))
  - Add support for dynamic shape in the TensorRT `layer_norm` plugin. ([#33448](https://github.com/PaddlePaddle/Paddle/pull/33448))
  - Add support for dynamic shape in TensorRT `hard_swish` plugin. ([#35214](https://github.com/PaddlePaddle/Paddle/pull/35214))
  - Add support for TensoRT `reduce_sum` and `gather_nd`. ([#33324](https://github.com/PaddlePaddle/Paddle/pull/33324))
  - Add support for int8 in TensorRT `qkv_context` plugin ([#34917](https://github.com/PaddlePaddle/Paddle/pull/34917), [#35504](https://github.com/PaddlePaddle/Paddle/pull/35504))
  - Add support for TensorRT conv3d. ([#35507](https://github.com/PaddlePaddle/Paddle/pull/35507))
  - Add support for broadcasting the input of the `multihead_matmul` fusion operator. ([#35780](https://github.com/PaddlePaddle/Paddle/pull/35780))
  - Inference supports for TensorRT8 sparse inference, with performance improved by 10%-30% for ERNIE model with variable-length input at different batch_sizes, and performance improved by 10% for ResNeXt101_32x4d model at different batch_sizes under test environment.  ([#36659](https://github.com/PaddlePaddle/Paddle/pull/36659))

- Nvidia Jetson native support enhancements
  - Add the Op support, for the Jetson Nano/TX2, two devices with lower arithmetic power. We made targeted optimizations. Now add the support for 17 OPs such as `pool2d`, `pool_max`, `conv3d_transpose`, etc. ([#35378](https://github.com/PaddlePaddle/Paddle/pull/35378))
  - For the Jetson Nano, we add new models: DPN68, EfficientNetB0, ttfnet, fcn_hrnetw18, hardnet. ([#35378](https://github.com/PaddlePaddle/Paddle/pull/35378))
  - For Jetson TX2, add new models: deeplabv3p_resnet50, deeplabv3_resnet50, fcn_hrnetw18, hardnet, pspnet, ttfnet, unet. ([#35378](https://github.com/PaddlePaddle/Paddle/pull/35378))


- Kunlun XPU interface feature extensions
  - Add the `set_xpu_device_id` interface to support setting the device number of the Kunlun chip in the inference ([#35572](https://github.com/PaddlePaddle/Paddle/pull/35572))
- In Inference python `copy_from_cpu` interface, add input type check. Report errors in advance for wrong type inputs.  ([#36552](https://github.com/PaddlePaddle/Paddle/pull/36552))

### **(3) Troubleshooting**

#### **Framework and API fixing**

- Operator fixing
  - Fix split op: when axis input is less than 0, address access error occurs when converting TensorRT. Filter out the cases that neither static nor dynamic shape is supported when axis is equal to 0. ([#35127](https://github.com/PaddlePaddle/Paddle/pull/35127))
  - Fix the bug where transpose static shape is wrong when axis is `[0, 1]`. ([#35138](https://github.com/PaddlePaddle/Paddle/pull/35138))
  - Fix the functional alignment of gather op with native paddle op, and improve op teller filtering conditions. ([#35784](https://github.com/PaddlePaddle/Paddle/pull/35784))
  - Fix int8 branching of fc op. ([#34787](https://github.com/PaddlePaddle/Paddle/pull/34787), [#32671](https://github.com/PaddlePaddle/Paddle/pull/32671))
  - Fix op teller filtering condition for reshape. ([#34787](https://github.com/PaddlePaddle/Paddle/pull/34787), [#34583](https://github.com/PaddlePaddle/Paddle/pull/34583))
  - Fix poor multi-threaded inference efficiency for recurrent op. ([#36053](https://github.com/PaddlePaddle/Paddle/pull/36053))
  - Fix the overflow bug of int values in gather and scatter op. ([#35544](https://github.com/PaddlePaddle/Paddle/pull/35544))
  - Fix the ctc op divide-by-zero error.  ([#34724](https://github.com/PaddlePaddle/Paddle/pull/34724))
  - Fix a crash caused by inserting a scale op when the model input contains a bool type. ([#35176](http://github.com/PaddlePaddle/Paddle/pull/35176))
  - Fix complex scaler and Tensor operations failure bug. ([#33699](https://github.com/PaddlePaddle/Paddle/pull/33699))

- Frame feature fixing
  - Fix an out-of-bounds GPU memory access bug when batching data for some ernie models. ([#35077](https://github.com/PaddlePaddle/Paddle/pull/35077))
  - Fix a possible accuracy bug in the running of the ernie model FP16 with precision. ([#34771](https://github.com/PaddlePaddle/Paddle/pull/34711))
  - Fix the incorrect output bug due to an inconsistent order of inputs when the ernie becomes longer. ([#33575](https://github.com/PaddlePaddle/Paddle/pull/33575))
  - Fix a bug where the allocator function is abnormal in multi-stream state. ([#32932](https://github.com/PaddlePaddle/Paddle/pull/33575))
- Fix a possible crash bug of ERNIE model under TRT8. ([#36769](https://github.com/PaddlePaddle/Paddle/pull/36769))
- Fix a bug of crash and accuracy when Pool and Slice are used. ([#36666](https://github.com/PaddlePaddle/Paddle/pull/36666))
- Fix an accuracy bug of yolo_box op caused by a wrong formula. ([#36365](https://github.com/PaddlePaddle/Paddle/pull/36365))
- Fix a bug where quantized matmul_v2 does not infer properly under TRT. ([#36821](https://github.com/PaddlePaddle/Paddle/pull/36821))
- Fix a bug where quantized op is incorrectly added when quantizing matmul_v2. ([#36820](https://github.com/PaddlePaddle/Paddle/pull/36820))
- Fix a bug with the operators batch_norm and elementwise_add reporting an error when TRT is enabled in 3D application scenarios. ([#36446](https://github.com/PaddlePaddle/Paddle/pull/36446))
- Fix a bug where the prediction model saved by the high-level linear api cannot not be optimized by Pass fusion. ([#36500](https://github.com/PaddlePaddle/Paddle/pull/36500))
- Fix the Pass of MatmulV2ToMul, re-qualify (matmul_v2 to mul) mapping pass, add Pass of MatmulV2ToMatmul, qualify (matmul_v2 to matmul) mapping pass condition (not supporting broadcast), and modify (matmul, mul) op_teller mapping condition.  ([#36652](https://github.com/PaddlePaddle/Paddle/pull/36652)

#### **Back-end capability fixing**

- TensorRT sub-graph engine fixing
  - Fix an out-of-bounds error reporting bug with slice plugin's ends parameter in the TensorRT dynamic shape. ([#35357](https://github.com/PaddlePaddle/Paddle/pull/35357))
  - Fix the bug of keepdim=false that is not supported when reduce op is converted to reduce_all = 1 of TensorRT. ([#35145](https://github.com/PaddlePaddle/Paddle/pull/35145))
  - Fix the decrease_axis parameter bug when slice op is converted to TensorRT. ([#35100](https://github.com/PaddlePaddle/Paddle/pull/35100))
  - Fix the bug that negative scale is not supported when nearest_interp op is converted to TensorRT dynamic shape.Fix a bug that scale has higher priority than outh and outw. ([#35405](https://github.com/PaddlePaddle/Paddle/pull/35405))
  - Fix the bug that padd op's paddings parameter is not the same as tensorrt. ([#35371](https://github.com/PaddlePaddle/Paddle/pull/35371))
  - Add the 4-dimensional padding support for conv2d op to converting to TensorRT. Filter the cases that the padding_algorithm is SAME and VALID when conv2d op is converted to TensorRT. ([#35627](https://github.com/PaddlePaddle/Paddle/pull/35627))
  - Add the handling of padding_algorithm as SAME for pool2d op converting to TensorRT. Filter the cases that ksize in exclusive mode less than or equal to padings. ([#35923](https://github.com/PaddlePaddle/Paddle/pull/35923))
  - Fix the bug of not supporting the Min and Max inputs when clip op is converted to TensorRT. ([#35694](https://github.com/PaddlePaddle/Paddle/pull/35694))
  - Fix the bug of not supporting the approximate attribute when gelu op is converted to TensorRT. ([#35529](https://github.com/PaddlePaddle/Paddle/pull/35529))
  - Fix the bug of not supporting the 2-dimensional inputs when affine_channel is converted to TensorRT. ([#35496](https://github.com/PaddlePaddle/Paddle/pull/35496))
  - Fix an unstable TensorRT sub-graph matching bug. ([#35147](https://github.com/PaddlePaddle/Paddle/pull/35147))
  - Fix the bug of the TensorRT engine not released after prediction engine destruction. ([#35842](https://github.com/PaddlePaddle/Paddle/pull/35842), [#35938](https://github.com/PaddlePaddle/Paddle/pull/35938))
  - Fix the bug of incorrect conversion to trt of the paddle operator in opaddle-trt static mode if the shape attribute batch dimension of reshape is -1. ([#34007](https://github.com/PaddlePaddle/Paddle/pull/34007))
  - Fix the bug of not supporting the RoisNum attribute when roi_align is converted to TensorRT. Fix the incorrect computation when aligned is True and Sampling_ratio = -1 in dynamic shape. ([#35549](https://github.com/PaddlePaddle/Paddle/pull/35549))
  - Fix the bug of not supporting the AxisTensor property when concat is converted to TensorRT. ([#35545](https://github.com/PaddlePaddle/Paddle/pull/35545))
  - Fix the bug of not supporting ScaleTensor property when scale is converted to TensorRT or not supporting 1-dimensional input in the static shape. ([#35225](https://github.com/PaddlePaddle/Paddle/pull/35225))
  - Fix the bug of not supporting the MomentumTensor property when batchnorm is converted to TensorRT. ([#35527](https://github.com/PaddlePaddle/Paddle/pull/35527))
  - Fix the bug of not supporting ShapeTensor when reshape is converted to TensorRT, fix the bug of not supporting the 1-Dimensional input in the Shape property and static shape.  ([#35166](https://github.com/PaddlePaddle/Paddle/pull/35166))
  - Add support for TensorRT tile operator. ([#34388](https://github.com/PaddlePaddle/Paddle/pull/34388))
  - Add support for TensorRT reduce mean operator. ([#34204](https://github.com/PaddlePaddle/Paddle/pull/34204))
  - Fix a possible crash when using gather op. ([#33999](https://github.com/PaddlePaddle/Paddle/pull/33999))
  - Fix a flag in TensorRT int8 incorrectly using debug (run only the int8 kernel, resulting in performance degradation). ([#34704](https://github.com/PaddlePaddle/Paddle/pull/34704))
  - Fix a computation error with gather_nd op when calling TensorRT on 2-dimensional inputs. ([#35464](https://github.com/PaddlePaddle/Paddle/pull/35464))
  - Fix hard_sigmoid op computation error when calling TensorRT with 2-dimensional input. ([#35908](https://github.com/PaddlePaddle/Paddle/pull/35908))
  - Fix computation error in prelu op when calling TensorRT on 2-dimensional inputs. ([#35512](https://github.com/PaddlePaddle/Paddle/pull/35512))
  - Fix a crash caused by using right slash as path separator in TensorRT inference on windows. ([#33853](http://github.com/PaddlePaddle/Paddle/pull/33853))


#### **Others**

- Fix the bug when pruning inverse operator script encounters an error with Chinese character comments. ([#33937](https://github.com/PaddlePaddle/Paddle/pull/33937), [#33919](https://github.com/PaddlePaddle/Paddle/pull/33919))
- Fix the bug of an error in compile-time single-test model download caused by incomplete single-test inference, add MD5 download validation for test model download. ([#33264](https://github.com/PaddlePaddle/Paddle/pull/33264), [#33217](https://github.com/PaddlePaddle/Paddle/pull/33217))
- Fix broadcast bug in blazeface model where mkldnn elementwise op is not supported.  ([#33549](https://github.com/PaddlePaddle/Paddle/pull/33549))
- Fix swin_transformer mkldnn inference error reporting bug. ([#35740](https://github.com/PaddlePaddle/Paddle/pull/35740))
- Fix the paddlex.deploy.Predictor oneDNN multi-threaded execution unet error reporting bug. ([#35231](https://github.com/PaddlePaddle/Paddle/pull/35231))
- Fix the bug with oneDNN setCacheCapacity not limiting memory. ([#33571](https://github.com/PaddlePaddle/Paddle/pull/33571))




## **Environment Adaptation**

### **Compiler Installation**

- For Windows, support `Ninja compilation build method`, compilation speed, ease of use, and stability. These features are improved greatly. Windows users can perform local source code compilation for Paddle via `pip install ninja`. ([#31161](https://github.com/PaddlePaddle/Paddle/pull/31161), [#31449](https://github.com/PaddlePaddle/Paddle/pull/31449), [#32987](https://github.com/PaddlePaddle/Paddle/pull/32987), [#33140](https://github.com/PaddlePaddle/Paddle/pull/33140), [#33155](https://github.com/PaddlePaddle/Paddle/pull/33155))
- Only python3.7 is kept in the release mirror. Remove python3.5, python3.6, python3.8, python3.9 and paddle packages of the corresponding python versions. The mirror size is reduced.The mirror size is reduced by 30%~50%. ([#32688](https://github.com/PaddlePaddle/Paddle/pull/32688))
- TensorRT library is used for inference. Only paddle training base functions in the release mirror are supported, without needing to support TensorRT.Delete the TensorRT library from the distribution image to prevent users from using the mirror by mistake. ([#34266](https://github.com/PaddlePaddle/Paddle/pull/34266))

### **New Hardware Adaptation**

- Support Hygon DCU chip training and inference. Support up to 9 classifications and 70 models. 
  - For Hygon DCU, add the support of 5 PaddleDetection models. 
  - For Hygon DCU, add the support for 6 PaddleGAN models.
  - For Hygon DCU, add the support for 13 PaddleSeg models.
  - For Hygon DCU, add the support for 3 PaddleNLP models.
  - For Hygon DCU, add the support for 4 PaddleOCR models.
  - For Hygon DCU, add the support for 3 PaddleVideo models.
- Support Kunlun 2nd generation chip (XPU-2) training. Support ResNet50, SSD, Bert, Transformer and many other models. Support static map + dynamic map training. Support mixed precision training.

## Thanks to our Contributors

This release contains contributions from:

0x45f, 123malin, Adam Osewski, Aganlengzi, Aurelius84, Baibaifan, Bo Liu, CheQiXiao, Chen Long, Chen Weihang, CtfGo, Double\_V, Ethanzjp, Fan Zhang, Feiyu Chan, Feng Xing, From00, GT-Zhang, Guanghua Yu, Guoxia Wang, Haipeng Wang, Hao Lin, Haohongxiang, Hui Zhang, Huihuang Zheng, HydrogenSulfate, IMMORTAL, JYChen, JZ-LIANG, Jacek Czaja, Jack Zhou, Jackwaterveg, Jeng Bai-Cheng, Jiangxinz, Jiaqi Liu, Jiawei Wang, JingZhuangzhuang, June Weng, Kaipeng Deng, Kqnonrime, LJQ❤️, Leo Chen, Li Min, LielinJiang, Lijunhui, Linjie Chen, Liu-xiandong, LiuWei, Ming-Xu Huang, MissPenguin, PaddlePM, Pei Yang, Peihan, Qi Li, QingshuChen, Ren Wei (任卫), Roc, Shang Zhizhou, ShenLiang, Shibo Tao, Siming Dai, Sing\_chan, TCChenLong, TTerror, TeslaZhao, Thomas Young, Thunderbrook, Tongxin Bai, WJJ1995, WangXi, Wangzheee, Wei Shengyu, WeiXin, Weilong Wu, Wenyu, Wilber, XGZhang, XYZ, XYZ916829, XiangGao, Xiaoxu Chen, YUNSHEN XIE, Yanxing Shi, Yiqun Liu, YuanRisheng, Yuang Liu, Yulong Ao, Zeng Jinle, Zhang Ting, Zhang Zheng, Zhanlue Yang, Zhen Wang, Zhong Hui, Zhou Wei, andreazanetti, andyjpaddle, arlesniak, baoachun, cc, ceci3, chajchaj, chenenquan, chenjian, chentianyu03, crystal, cuicheng01, danleifeng, denglin-github, duanboqiang, dyning, feng626, feng_shuai, furnace, gongweibao, heliqi, hlygit66666, hong, hong19860320, houj04, huangjun12, huangxu96, huzhiqiang, iducn, jakpiase, jiangcheng, joanna.wozna.intel, jzhang533, kuizhiqing, levi131, lidanqing, lilong12, limingshu, littletomatodonkey, liu zhengxi, liutiexing, liuyuhui, liym27, lyuwenyu, lzzyzlbb, niuliling123, pangyoki, parap1uie-s, ronnywang, root, seemingwang, shangliang Xu, shiyutang, smallv0221, sunli, sunzhongkai588, taixiurong, tangwei12, tianshuo78520a, veyron95, wangguanqun, wangguanzhong, wanghuancoder, wangna11BD, wangxinxin08, wangzhen38, wangzhuang01, wawltor, wenbin, whs, will-jl944, wuhuachaocoding, wuhuanzhou, xiaoting, xiaoxiaohehe001, xiayanming, xiegegege, xiemoyuan, xiongkun, yaoxuefeng, yeliang2258, yingyibiao, zhangbo9674, zhangchunle, zhangkaihuo, zhaoyingli, zhiboniu, zhoujun, zhouzj, zhulei, zhupengyang, zlsh80826, zmx, zyfncg, 李季, 津, 王明冬, 石晓伟