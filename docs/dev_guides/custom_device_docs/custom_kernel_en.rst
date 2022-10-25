####################
Custom Kernel
####################

The custom kernel is the implementation of corresponding operators of the kernel function (or kernel). The PaddlePaddle framework provides the custom kernel for the external device registered by the custom runtime, achieving the compiling, registration, and automatic loading of the kernel independent of the framework.
The implementation of the custom kernel is based on the public kernel statement of PaddlePaddle, and public C++ API and register macro.


- `Kernel function statement <./custom_kernel_docs/kernel_declare_en.html>`_ : to introduce the kernel statement of PaddlePaddle
- `Kernel implementation API <./custom_kernel_docs/cpp_api_en.html>`_ : to introduce the C++ API required in the implementation of the custom function.
- `Kernel register API <./custom_kernel_docs/register_api_en.html>`_ : to introduce the register macro of the custom kernel.


..  toctree::
    :hidden:

    custom_kernel_docs/kernel_declare_en.md
    custom_kernel_docs/cpp_api_en.rst
    custom_kernel_docs/register_api_en.md
