#############
Kernel Primitives API
#############

This part provides advanced developers with CUDA Kernel Primitive API to accelerate kernel development. These APIs are provided to the block. Developers can directly pass in the corresponding data pointer and operation type to complete the corresponding calculation. Currently, only global data pointers and register pointers are supported. To help users develop high-performance operators, we provide IO APIs with high memory access efficiency, which can help developers gain better performance while improving development efficiency.

API was divided into IO API and Compute API:

1. IO API: A high-performance data read and write API designed in conjunction with GPU hardware features.

2. Compute API: Design general calculation functions based on GPU computing characteristics, such as ELementwiseBinary, ElementwiseUnary, etc.

API List
############

+--------------------------------------+------------------------------------------------------------ +
| API Name                             |  Functions                                                  |
+======================================+=============================================================+
| ReadData                             | IO, Efficiently read data from global memory into registers.|
+--------------------------------------+-------------------------------------------------------------+
| ReadDataBc                           | IO, for Broadcast op. Calculate the original input data     |
|                                      | coordinates corresponding to the output according to the    |
|                                      | data offset of the current block and the original input     |
|                                      | pointer, and read the input data into the register.         |
+--------------------------------------+-------------------------------------------------------------+
| ReadDataReduce                       | IO, for Reduce op. According to the Reduce configuration,   |
|                                      | calculate the data coordinates that require Reduce, and read|
|                                      | the data into the register.                                 |
+--------------------------------------+-------------------------------------------------------------+
| WriteData                            | IO. Efficiently write data from registers to global memory. |
+--------------------------------------+------------------------------------------------------------ +
| ElementwiseUnary                     | Compute API. Unary calculation API. Complete unary function |
|                                      | operations according to OpFunc.                             |
+--------------------------------------+-------------------------------------------------------------+
| ELementwiseBinary                    | Compute API. The inputs and output have the same shape.     |
|                                      | Complete the binary function operation according to the     |
|                                      | OpFunc.                                                     |
+--------------------------------------+-------------------------------------------------------------+
| ELementwiseTernary                   | Compute API，The inputs and output have the same shape.     |
|                                      | Complete the ternary functions operation according to the   |
|                                      | OpFunc.                                                     |
+--------------------------------------+-------------------------------------------------------------+
| ELementwiseAny                       | Compute API，The inputs and output have the same shape.     |
|                                      | Complete the compute according to the OpFunc.               |
+--------------------------------------+-------------------------------------------------------------+
| CycleBinary                          | Compute API. Input 1 and input 2 have different shapes,     |
|                                      | input 2 and output have the same shape, and complete the    |
|                                      | binary loop calculation according to OpFunc.                |
+--------------------------------------+-------------------------------------------------------------+
| Reduce                               | Compute API, Complete the reduce calculation in the block or|
|                                      | in the thread according to the reduce mode.                 |
+--------------------------------------+-------------------------------------------------------------+

API Description
##############

- `IO API <./io_api_en.html>`_ : Describes the definition and functions of IO APIs.
- `Compute API <./compute_api_en.html>`_ : Describes the definition and functions of compute APIs.

API Examples
############

- `elementwise_add <./elementwise_add_case_en.html>`_ : Addition operation, the input and output shapes are the same.
- `reduce <./reduce_case_en.html>`_ : Only the highest dimension is involved in reduce.

..  toctree::
    :hidden:


    io_api_en.md
    compute_api_en.md
    elementwise_add_case_en.md
    reduce_case_en.md
