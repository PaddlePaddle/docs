#####################
Kernel Primitive API
#####################

The Kernel Primitive API abstracts and encapsulates the underlying code in the implementation of operator kernel functions, providing high-performance block-level IO operation and computation operation. Using the Kernel Primitive API for Kernel development can focus more on the implementation of computing logic, greatly reduce the amount of code while ensuring performance, and realize the decoupling of operator computing and hardware.

This part provides PaddlePaddle senior developers with CUDA Kernel Primitive API for Kernel development. This type of API can help developers gain better performance while improving development efficiency. Kernel Primitive API mainly includes IO API, Compute API and OpFunc. IO API can efficiently complete data reading and writing operations between global memory and registers; Compute API is a general calculation function, such as ElementwiseBinary, ElementwiseUnary, etc.; OpFunc is used to define Compute API. For example, to implement the Add operation, you need to define AddFunctor for ElementwiseBinary calls. Developers can directly use the default OpFunc or customize them as needed. The specific implementation rules will be described in detail in the OpFunc section. The current APIs are all block-level multi-threaded APIs. Developers can directly pass in the data pointer and operation type of the current block to complete the corresponding calculations. Currently, only global data pointers and register pointers are supported.


API List
############

+--------------------------------------+-------------------------------------------------------------+
| API Name                             | Descriptions                                                |
+======================================+=============================================================+
| ReadData                             | IO, read data from global memory into registers.            |
+--------------------------------------+-------------------------------------------------------------+
| ReadDataBc                           | IO, read data in Broadcast form, calculate the input        |
|                                      | coordinates according to the data offset of the current     |
|                                      | block and the original input pointer, and read the input    |
|                                      | data into the register.                                     |
+--------------------------------------+-------------------------------------------------------------+
| ReadDataReduce                       | IO, read data in Reduce form, read the input data from      |
|                                      | global memory into register for Reduce.                     |
+--------------------------------------+-------------------------------------------------------------+
| WriteData                            | IO, write data from registers to global memory.             |
+--------------------------------------+-------------------------------------------------------------+
| ElementwiseUnary                     | Compute API, the unary computing API with the same input and|
|                                      | output Shape, completes unary function operations according |
|                                      | to OpFunc calculation rules.                                |
+--------------------------------------+-------------------------------------------------------------+
| ElementwiseBinary                    | Compute API. The inputs and output have the same shape.     |
|                                      | Complete the binary function operation according to the     |
|                                      | OpFunc.                                                     |
+--------------------------------------+-------------------------------------------------------------+
| ElementwiseTernary                   | Compute API，The inputs and output have the same shape.     |
|                                      | Complete the ternary functions operation according to the   |
|                                      | OpFunc.                                                     |
+--------------------------------------+-------------------------------------------------------------+
| ElementwiseAny                       | Compute API，The inputs and output have the same shape.     |
|                                      | Complete the compute according to the OpFunc.               |
+--------------------------------------+-------------------------------------------------------------+
| CycleBinary                          | Compute API. Input1 and input2 have different shapes,       |
|                                      | input2 and output have the same shape, and complete the     |
|                                      | binary loop calculation according to OpFunc.                |
+--------------------------------------+-------------------------------------------------------------+
| Reduce                               | Compute API, Complete the reduce calculation according to   |
|                                      | the reduce mode.                                            |
+--------------------------------------+-------------------------------------------------------------+

OpFunc List
############

+--------------------------------------+-------------------------------------------------------+
| Functor name                         | Descriptions                                          |
+======================================+=======================================================+
| ExpFunctor                           | Unary Functor, performs Exp operation.                |
+--------------------------------------+-------------------------------------------------------+
| IdentityFunctor                      | Unary Functor, which performs type conversion on input|
|                                      | data.                                                 |
+--------------------------------------+-------------------------------------------------------+
| DivideFunctor                        | Unary Functor, returns the division result of the     |
|                                      | input.                                                |
+--------------------------------------+-------------------------------------------------------+
| SquareFunctor                        | Unary Functor, returns the square of the data.        |
+--------------------------------------+-------------------------------------------------------+
| MinFunctor                           | Binary Functor, returns the smallest value among the  |
|                                      | inputs.                                               |
+--------------------------------------+-------------------------------------------------------+
| MaxFunctor                           | Binary Functor, returns the maximum value of the      |
|                                      | inputs.                                               |
+--------------------------------------+-------------------------------------------------------+
| AddFunctor                           | Binary Functor, returns the sum of the inputs.        |
+--------------------------------------+-------------------------------------------------------+
| MulFunctor                           | Binary Functor, returns the product of the inputs.    |
+--------------------------------------+-------------------------------------------------------+
| LogicalOrFunctor                     | Binary Functor, which returns the logical or result of|
|                                      | the inputs.                                           |
+--------------------------------------+-------------------------------------------------------+
| LogicalAndFunctor                    | Binary Functor, which returns the logical and result  |
|                                      | of the inputs.                                        |
+--------------------------------------+-------------------------------------------------------+
| DivFunctor                           | Binary Functor, returns the result of division of the |
|                                      | inputs.                                               |
+--------------------------------------+-------------------------------------------------------+
| FloorDivFunctor                      | Binary Functor, returns the result of division of the |
|                                      | inputs.                                               |
+--------------------------------------+-------------------------------------------------------+


API Description
###############

- `IO API <./io_api_en.html>`_ : Describes the definition and functions of IO APIs.
- `Compute API <./compute_api_en.html>`_ : Describes the definition and functions of compute APIs.
- `OpFunc <./functor_api_en.html>`_ : Introduce the Functors provided by the Kernel Primitive API.

API Examples
############

- `ElememtwiseAdd <./add_example_en.html>`_ : Addition operation, the input and output shapes are the same.
- `Reduce <./reduce_example_en.html>`_ : Only the highest dimension is involved in reduce.

..  toctree::
    :hidden:


    api_description_en.rst
    example_en.rst
