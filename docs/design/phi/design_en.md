# Paddle HIgh reusability operator library (PHI) Design Document

Paddle HIgh reusability operator library (PHI), or we also call it 'functional operator library', supports to implement new operator kernels based on existing operator kernel functions and 'Kernel Primitives API (KPS)', and supports plug-in access to new hardware or new acceleration library.

In order to solve the problems of unclear operator interface in the original operator library of the Paddle Fluid Framework, high cost of operator reuse, and poor scheduling performance, we refactored the operator library of the Paddle Framework, designed flexible and efficient functional paradigm.

The operator library PHI can implement new operators by combining calls to functional operator interfaces. The new operator library provides more than 200 C++ operation APIs that are consistent with the Python development interface, and nearly 500 forward and reverse functional operator kernels that can be combined and called, which can greatly reduce the development cost of native operator and custom operator.

## 1. Background and Objectives

> Introduce the problems to be solved in designing and building the PHI operator library


The PHI operator library project was initially launched to support the refactoring of the paddle dynamic graph architecture to reduce scheduling overhead and improve the reuse capability of OpKernel development. However, the subsequent decision to take this opportunity to establish an operator library that can be used in both training and inference scenarios (including server-side and mobile-side scenarios), reduce the cost of infrastructure development and operator maintenance in the paddle ecosystem in the long run, so we expanded the target scope of the project. At present, PHI has carried a number of goals.

### 1.1 Background issues

Specifically, the PHI operator library project carries the expectation to solve the following problems of Paddle.

#### 1.1.1 Poor reusability between Op&OpKernel

Before version 2.3, the reusability between Operators (Op) in Paddle was relatively poor. Only in a few grad Ops, some simple Ops were reused by calling SetType in the GradOpMaker implementation. In most cases where the existing Op implementation can be reused, the code is rewritten by copy..

The root cause of poor reusability is the inflexibility of the original Op system design:

1. When an Op reuses the `Opkernel::Compute` method of another Op, an `ExecutionContext` needs to be constructed first, and the reuse method is relatively cumbersome

    > It will be much more convenient if you can directly call the Kernel in the form of a function

2. Due to the overhead introduced by additional data structure construction and independent Op scheduling, from the perspective of computing performance, it is better to copy the calculation code directly when reusing Op, which leads us to gradually abandon the earlier principle of grad Op reusing forward Op, and began to implement Kernel separately for each grad Op, so that Paddle maintains a large number of grad OpKernel implementation codes internally.

    > Only when the overhead of reusing Ops is small enough, reusing existing Ops to implement new Ops can be widely promoted

#### 1.1.2 Conciseness and fine-grained execution scheduling

##### 1.1.2.1 Dynamic graph

After the release of Paddle 2.0, it has received many feedbacks from internal and external users that the performance of the dynamic graph is several times lower than that of competing products in the execution scenario of small model on CPU.

The main reason for this problem is: the execution path of the C++ side of the Padddle dynamic graph is relatively long and the scheduling overhead is relatively heavy, which is related to the early design of the dynamic graph which is compatible with the static graph and inherits many object construction processes of the static graph Op.

- Issue issue: https://github.com/PaddlePaddle/Paddle/issues/28774

Therefore, the dynamic graph needs to be upgraded to a function-based scheduling architecture, and this problem can be solved by abandoning the original complex Op system, which depends on the OpKernel being changed to a functional writing method.

##### 1.1.2.2 Static image + IR

Our current static graph mode are not "static" enough. At present, static graph mode still have a lot of logic for dynamic selection at runtime, for example, selecting OpKernel at runtime, judging whether to copy data across devices at runtime, etc.. However, these can actually be determined during the compilation of the static graph mode network, and the execution process is determined as a series of OpKernel executions, and no dynamic judgment selection is made, thereby further improving the execution efficiency.

And these rely on the fine-grained OpKernel itself, decoupling the existing complex large OpKernel into small Kernels for specific scenarios and specific devices.

#### 1.1.3 Ease of use improvement requirements for custom operators

The new custom C++ external operator paradigm released in early 2021 has a relatively intuitive usage at the level of interface and function writing, but because we lack the C++ APIs for basic operations, in fact, when implementing specific custom Op operation logic, such as basic addition, subtraction, multiplication and division and matrix operations, still need to be reimplemented again, and Paddle's existing and optimized basic operations cannot be reused, development costs are still relatively high. In order to reuse the basic operations inside Paddle, the Op paradigm must be upgraded to functional paradigm, and build the corresponding C++ API system.

#### 1.1.4 Build an integrated training and inference operator library to reduce the maintenance cost of inference operators

For a long time, because the Paddle and Paddle-Lite operators are maintained separately, the new paddle operator, if Paddle-Lite needs it, must be manually reimplemented in Paddle-Lite, and when the Paddle operator is upgraded, Paddle-Lite does not perceive it in time, which will directly lead to bugs in the inference model when lite is executed, which introduces high maintenance costs. Only a unified operator library can solve this problem for a long time.

Therefore, this functional operator library will be jointly constructed by training and inference team, and will serve as an independent compilation component and underlying infrastructure (not yet independently split), which can serve training, prediction, and Lite execution systems at the same time.

#### 1.1.5 The adaptation of the new inference Runtime design 'infrt'

Inference team designed a new runtime 'infrt'. It is expected to unify the execution system of Paddle-Inference and Paddle-Lite. It is necessary to directly call the operators in the PHI operator library jointly built this time. Therefore, the adaptation to 'infrt' needs to be considered in the design. (Currently the 'infrt' project is temporarily on hold).

#### 1.1.6 Op and Kernel parameter normalization

The Python 2.0 API project in 2020 standardized the argument list of the Paddle Python-side API, making it concise, easy to use, and standard. However, due to cost considerations, the argument list at the Op level was not standardized, so there will be many early developed operators that differ greatly in arguments from the Python API. For example, conv op, the Python API has only 8 arguments, but the corresponding C++ Conv Op has 30+ arguments. API and Op are essentially the same layer of concepts, both are descriptions of an operation, and the arguments should be consistent. In order to solve this problem, 'the operator definition enhancement project' was launched, and the declarations of 'AsExtra' and 'AsQuant' were added to some unnecessary arguments, but the problem was not fundamentally solved, which is what the construction of the PHI operator library hopes to solve.

We hope to be able to achieve the same three-layer arguments of Python API -> Op(C++ API) -> Kernel API, so that the overall structure is clear, and the reuse relationship of each layer is clear enough. Maintaining a set of official Python API documents can basically satisfy the common reference requirements of the three-tier API, no longer focus on maintaining additional document systems and reduce maintenance costs.

### 1.2 Objectives and Scope

- Overall goal: The core framework of Paddle reuses the same functional operator library; the basic data structure Tensor has good scalability, and fundamentally achieves consistent training and inference; the basic components are stable and reliable, and the incremental development experience is good.

- Target range:

  - The initial construction of the PHI operator library paid more attention to Kernel "migration". Due to the consideration of time and labor costs, the original OpKernel logic migration is not forced to be upgraded to "combined" writing for the time being, and the same is true for the forward and gradient Kernels
  - The "combined Kernel extension development" capability provided by the PHI operator library initially serves the new operators of subsequent increments, and the existing operators still maintain their original coding implementation, reducing the cost of migration
  - The "new hardware expansion capability" provided by the PHI operator library is initially only provided within the scope of the new hardware itself. For example, the XPU has implemented 50 Kernels, and then it can combine new Kernels based on 50 Kernels, but this is only limited to the XPU Within the scope, its implementation is not common with CPU, CUDA, etc.
  - The PHI operator library project focuses on the work of "Kernel functionalization & Op normalization", Kernel is changed to functional format, C++ API and Op naming and arguemnts list are gradually normalized to Python API under the premise of ensuring compatibility as much as possible


## 2. Design Overview

### 2.1 Naming and Location

The PHI code directory is inside the paddle directory, which is at the same level as fluid, rather than inside the fluid directory. Phi is a basic component that is called by various upper-level runtimes such as fluid, lite, and infrt, and it will be used later as a separately compiled dynamic library, therefore PHI is not suitable as the submodule of fluid.

### 2.2 Directory Structure

#### 2.2.1 Requirements of directory structure design

Training and inference require a clear operator library directory structure:

- The directory design should support various split compilation requirements of the operator library, which including: 

    - Split and compile according to the computing device.
        - For example, compile for cpu only, or compile for gpu only.
    - Split and compile according to the training and inference scenarios.
        - For example, the inference scenario does not compile backward-relevant kernels, nor forward kernels with Intermediate outputs
    - Precisely crop and compile according to the operators actually used by the mobile device (not supported yet)【prune or crop or clip?】
        - For example, a model uses `add` and `mul` only, ideally it could be cropped to only 2 kernels.

- In the long run, support the requirement of easily reusing kernel implementation.
    - Explanation: When reusing the kernel, the corresponding function implementation should be introduced through `include` easily, rather than cannot find the kernel because of the complex directory structure.
    
- In the long run, support the requirement of the unified writing method among cross-device kernels, and the writing method is intuitive and easy to use, without introducing unnecessary template parameters.
    - Explanation: Kernel Primitive API module is at the lower level of the operator library. Its long-term vision is that each operation uses only one kernel to adapt to various devices, the code that truly distinguishes the device is only in the implementation of the Kernel Primitive API. In the future, the template parameters should be limited to as concise as possible when passing complex parameters into the reused kernel.

- In terms of ease of use, developers can accurately understand where the newly added kernel should be placed, without ambiguity.
    - Explanation: When developers add an API, they will not be confused about which directory they should put the corresponding kernel in. Moreover, different people should have no ambiguous understanding of where the same kernel should be placed. 

- Do not introduce a lot of duplicate directory design.
    - Explanation: Concept splitting is needed, but also with boundaries. Avoid subdirectories with the same name occurring in multiple directories. For example, if `eigen`, `funcs`, `math` directories are placed under the cpu directory, then they shouldn't be placed under the gpu directory. The directory design of the new operator library is mainly divided according to the device, and the directory splitting at other levels should be weakened as much as possible. For example, try not to split based on functions, try not to split based on fields, etc.

- No file bloat during migration.
    - Explanation: Splitting kernels according to devices shouldn't cause a large-scale increase of kernel implementation files.

- Do not introduce too deep directory design.
    - Explanation: The directory level should not be too deep, otherwise it will lead to higher understanding and maintenance costs.

- Do not introduce excessive migration costs.
    - Explanation: When migrating the kernel, do not make too many changes to the kernel itself, otherwise the migration cost will be too high.

#### 2.2.2 Directory design details

##### 2.2.2.1 First level directory

```
paddle/phi
./api (High-level API exposed to the outside and corresponding implementation)
    ./include（High-level API header file exposed to the outside）
    ./lib（API implementation exposed to the outside）
./capi (C API exposed to the outside and corresponding implementation)
    ./include
    ./lib
./common (Basic data structures used both internally and externally)
./core (Basic components, such as basic Tensor-related interfaces, kernel registration interfaces, management units, etc.)
./backends (Basic components of each device and backend, including backend directories such as cpu, gpu, etc.)
./infermeta (Derivation functions for meta information such as shape, dtype, layout, etc.)
./kernels (Kernel implementation of each device and backend)
./ops (The definition of each Op, most of the work is done automatically by code generation in the future, and currently there are only compatible codes)
./tests (Unit test)
```

Some directory structure description:

- `api`: API module for external users.
    - Directly use the Python-like C++ Tensor computing API, which is highly consistent with the Python side.
    - This part may reversely depend on the framework's DeviceContextPool and other implementations, so it is managed separately.
    - On such APIs, training and prediction may also be different.
- `capi`: C API module, currently mainly serving the plug-in hardware access function.
- `common`: Data structures to be used both inside phi and in the PHI api directory.【疑问】 These data structures neither belong to the PHI core nor the api directory.
- `core`: PHI has some public module implementations that it needs, such as DenseTensor, kernel registration and management modules.
- `backends`: The backends include data structures that need to be added for each backend, such as CPUContext, GPUContext, etc.
    - The basic data structures are placed in the `core`, while the dedicated data structures of specific backends are not placed in the `core`, and the dependencies strictly ensure that the backends depend on the `core`, but the `core` cannot depend on the backends.【专有词汇用``】
    - Example 1: If Context is a base class, then put it in `core`, inherited CPUContext is in backends/cpu and GPUContext is in backends/gpu.
    - Example 2: TensorBase is in `core`, DenseTensor is used by most devices so that it is also in the `core`. If there is ONEDNNTensor, which is only used for ONEDNN, then it should be placed in backends/onednn.
- `infermeta`: The location of the infermeta function, the infermeta function is equivalent to infershape + inferdtype + inferlayout, etc.
- `kernels`: Kernels related to each device.
    - `cpu, gpu, ...`
- `ops`: Ops includes new forms of Op definitions, as well as some components compatible with original Ops.


##### 2.2.2.2 Kernels directory

```
paddle/phi/kernels
./ (Device-independent kernel declarations and implementations.)
./cpu (Include the kernel implementation of the cpu backend only.)
./gpu
./xpu
./onednn
./gpudnn
./impl (Considering the current situation, for easy reuse, this directory includes the consistent implementation of the original Kernel on CPU, GPU and other devices.)
./funcs (Including some functor and funcs that support multiple devices under the original fluid operators.)
./primitive (Includes basic implementation of the Kernel Primitive API)
...
```

The directory structure is described as follows:

- The main directory under kernels includes device-independent kernel.h and kernel.cc. In principle, each kernel has one .h and .cc
    - For example, if a kernel is implemented using Primitive api, or is implemented by reusing other basic kernels, there should be only one implementation for all devices, so its declaration and implementation can be placed directly in the kernels directory. (This is the ideal state in the future.)
    - At present, most of our kernels do not have the feature of unity implementation across devices, but the input parameters and return values of the kernel should be consistent except for `DeviceContext`, so the kernel parameter declaration header file is also placed in the current directory (consistent with the original design, `DeviceContext` and `T` are used as template parameters), The functions implementation of each device are placed in the corresponding device folder.
        - Note that the unity implementation across devices here does not mean that the CPU and GPU implementations of a kernel are unified, but the implementations of all devices are the same. Currently, it includes at least CPU, GPU, XPU, ONEDNN, GPUDNN, etc.
    - If the backward kernel does not need to support cropping, it can be merged appropriately (but if you want to leave the possibility of supporting end-to-side training, the backward kernel may also be a potential target for cropping)
- The next-level subdirectory of kernels, in principle, is created according to the backend classification, and only two special directories are reserved:
    - funcs: In order to be compatible with the directories of functor and function in the original fluid/operators directory, when placing functions and functors that support multiple backends, we organize them according to the original design that one header file corresponding to multiple .cc(u) (This part of the code may be removed in the future, because it will be gradually replaced by Kernel Primirive API and reuse between Kernels, so no over-design here.)
        - 
    