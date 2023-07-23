## [ torch 参数更多 ]torch.utils.cpp_extension.load
### [torch.utils.cpp_extension.load](https://pytorch.org/docs/stable/cpp_extension.html?highlight=torch+utils+cpp_extension+load#torch.utils.cpp_extension.load)

```python
torch.utils.cpp_extension.load(name,
                            sources,
                            extra_cflags=None,
                            extra_cuda_cflags=None,
                            extra_ldflags=None,
                            extra_include_paths=None,
                            build_directory=None,
                            verbose=False,
                            with_cuda=None,
                            is_python_module=True,
                            is_standalone=False,
                            keep_intermediates=True)
```

### [paddle.utils.cpp_extension.load](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/utils/cpp_extension/load_cn.html)

```python
paddle.utils.cpp_extension.load(name,
                            sources,
                            extra_cxx_cflags=None,
                            extra_cuda_cflags=None,
                            extra_ldflags=None,
                            extra_include_paths=None,
                            build_directory=None,
                            verbose=False)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch                     | PaddlePaddle            | 备注                                                   |
| -------------               | ------------            | ------------------------------------------------------ |
| name                        | name                    |  用于指定编译自定义 OP 时，生成的动态链接库的名字。                                    |
| sources                     | sources                 |   用于指定自定义 OP 对应的源码文件。                           |
| extra_cflags          | extra_cxx_cflags        |   用于指定编译 cuda 源文件时额外的编译选项。          |
| extra_cuda_cflags          | extra_cuda_cflags    |         用于指定编译 cuda 源文件时额外的编译选项。                      |
| extra_ldflags                 |extra_ldflags         |  用于指定编译自定义 OP 时额外的链接选项。                                 |
| extra_include_paths    | extra_include_paths   |  用于指定编译 cpp 或 cuda 源文件时，额外的头文件搜索目录。                               |
| build_directory       | build_directory       |    用于指定存放生成动态链接库的目录。                                    |
| verbose                 | verbose             | 用于指定是否需要输出编译过程中的日志信息，默认为 False。   |
| with_cuda                 | -                          | 决定是否将 CUDA 头文件和库添加到 build。 PaddlePaddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
| is_python_module          | -                          | 默认为 True，将生成的共享库作为 Python 模块导入，PaddlePaddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
| is_standalone             | -                          | 默认为 False，将构建的扩展作为一个普通的动态库加载到进程中，如果是 True，则构建一个独立的可执行文件，PaddlePaddle 无此参数，一般对网络训练结果影响不大，可直接删除。|
| keep_intermediates        | -                          | 默认为 True，保留中间文件，PaddlePaddle 无此参数，一般对网络训练结果影响不大，可直接删除。|
