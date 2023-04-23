#############################
Kernel Implementation APIs
#############################

The custom kernel-function implementation mainly depends on two parts: 1.APIs released by PaddlePaddle, including the context API, the tensor API, and the exception API; 2. APIs of the device encapsulation library. And the C++ API of PaddlePaddle has been released by the header file.


- `Context API <./context_api_en.html>`_ : about the C++ API of the device context
- `Tensor API <./tensor_api_en.html>`_ : about the C++ API of Tensor
- `Exception API <./exception_api_en.html>`_ : about the C++ API of exception handling


Note：There are abundant C++ API of PaddlePaddle. Three APIs will be introduced here and related classes and documents listed in corresponding websites are provided for developers.

..  toctree::
    :hidden:

    context_api_en.md
    tensor_api_en.md
    exception_api_en.md
