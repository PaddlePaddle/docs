# Build mobile inference library with minimum size

In the mobile application, there usually are some limitations to the size of an executable file.
Here we explore how to compile the inference library for minimum size.

**Note:**  
In the original PaddlePaddle, all computationally relevant code is implemented in Matrix.cpp and BaseMatrix.cu,
this causes the large compiled Matrix.o and BaseMatrix.o files which can not be split.
The module of Layer can be split, but the Layer forward and backward computing is included in the same file.
The configuration definition in proto is redundant. These all will lead to the size of inference library larger.

The new PaddlePaddle is being refactored, the calculation is based on Operator, the configuration description in proto is simplified.
These will bring a good optimization to the inference library size.

## How to build with minimum size
Here we mainly introduce some optimization in the compilation process to reduce the size of the inference library. These methods are also used in the refactored PaddlePaddle.

- Compile the code with `-Os` option: By specifying the build type(CMAKE_BUILD_TYPE) to MinSizeRel, the `-Os` option is used by the compiler during the build for minimum size code.
- Use protobuf-lite instead of protobuf: With the `--cpp_out lite` option, the generated proto configuration rely on MessageLite instead of Message,
so only libprotobuf-lite is needed for the link, and the size of the final executable file can be reduced.
- Use `--whole-archive` link option: In `libpaddle_capi_layers.a` contains all the object file of the layers, need use the `--whole-archive` option to ensure that all layers be linked to the executable file.
But don't forget to use `-no-whole-archive` after `libpaddle_capi_layers.a`, avoid this option affect other libraries.
[Here's an example](https://github.com/PaddlePaddle/Mobile/blob/develop/benchmark/tool/C/CMakeLists.txt#L41)
- Remove useless symbols in the shared library: When building a shared library by `libpaddle_capi_layers.a` and `libpaddle_capi_engine.a`,
you can remove the useless export symbols with the `--version-script` option to reduce the size of the `.dynsym` and `.dynstr` sections.
[Here's an example](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/capi/CMakeLists.txt#L61)
