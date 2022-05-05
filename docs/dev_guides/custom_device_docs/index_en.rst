#############################
Custom Device Access Guide
#############################

The custom device access decouples the framework from the device and makes it available to extend the backend of the PaddlePaddle device via plug-ins. In this way, developers can make a plug-in for PaddlePaddle only by implementing the standard API and compiling it into a dynamic-link library, instead of by modifying the code of PaddlePaddle. Now it is easier to develop hardware backends for PaddlePaddle.

The custom device access is composed of custom runtime and custom kernel. With the two modules, users can connect new custom devices to PaddlePaddle according to their own needs.

- `Custom Runtime <./custom_runtime_en.html>`_ : Introduction of custom runtime of the PaddlePaddle framework
- `Custom Kernel <./custom_kernel_en.html>`_ : Introduction of custom kernel of the PaddlePaddle framework
- `Example of Device Access <./custom_device_example_en.html>`_ : To demonstrate how to connect new custom devices to PaddlePaddle

..  toctree::
    :hidden:


    custom_runtime_en.rst
    custom_kernel_en.rst
    custom_device_example_en.md
