#############################
Custom Device Support
#############################

The custom device function decouples the framework from the device and makes it available to extend the backend of the PaddlePaddle device via plug-ins. In this way, developers can make a plug-in for PaddlePaddle only by implementing the standard API and compiling it into a dynamic-link library, instead of by modifying the code of PaddlePaddle. Now it is easier to develop hardware backends for PaddlePaddle.

The custom device function is composed of custom runtime and custom kernel. With the two modules, users can connect new custom devices to PaddlePaddle according to their own needs.

- `Custom Runtime <./custom_runtime_en.html>`_ : Introduction of custom runtime of the PaddlePaddle framework
- `Custom Kernel <./custom_kernel_en.html>`_ : Introduction of custom kernel of the PaddlePaddle framework
- `CustomDevice Example <./custom_device_example_en.html>`_ : The tutorial of add a new custom device to PaddlePaddle

..  toctree::
    :hidden:


    custom_runtime_en.rst
    custom_kernel_en.rst
    custom_device_example_en.md
