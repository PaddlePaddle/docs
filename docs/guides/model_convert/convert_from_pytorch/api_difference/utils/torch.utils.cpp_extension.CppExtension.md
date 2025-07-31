## [ torch 参数更多 ]torch.utils.cpp_extension.CppExtension
### [torch.utils.cpp_extension.CppExtension](https://pytorch.org/docs/stable/cpp_extension.html?highlight=torch+utils+cpp_extension+cppextension#torch.utils.cpp_extension.CppExtension)

```python
torch.utils.cpp_extension.CppExtension(name,
                                    sources,
                                    *args,
                                    **kwargs)
```

### [paddle.utils.cpp_extension.CppExtension](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/utils/cpp_extension/CppExtension_cn.html)

```python
paddle.utils.cpp_extension.CppExtension(sources,
                                    *args,
                                    **kwargs)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| name          | name            | 用于指定自定义 OP 对应的名字，Paddle 在不传入时可以自动推导。  |
| sources         | sources         | 用于指定自定义 OP 对应的源码文件。   |
|*args         | *args          |   用于指定 Extension 的其他参数，支持的参数与 setuptools.Extension 一致。 |
| **kwargs      | **kwargs        |   用于指定 Extension 的其他参数，支持的参数与 setuptools.Extension 一致。 |
