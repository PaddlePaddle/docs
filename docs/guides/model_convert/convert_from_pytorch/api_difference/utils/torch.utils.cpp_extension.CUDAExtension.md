## [ torch 参数更多 ]torch.utils.cpp_extension.CUDAExtension
### [torch.utils.cpp_extension.CUDAExtension](https://pytorch.org/docs/1.13/cpp_extension.html?highlight=torch+utils+cpp_extension+cudaextension#torch.utils.cpp_extension.CUDAExtension)

```python
torch.utils.cpp_extension.CUDAExtension(name,
                                    sources,
                                    *args,
                                    **kwargs)
```

### [paddle.utils.cpp_extension.CUDAExtension](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/utils/cpp_extension/CUDAExtension_cn.html)

```python
paddle.utils.cpp_extension.CUDAExtension(sources,
                                    *args,
                                    **kwargs)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| name          | -            | 参数 name，PaddlePaddle 无此参数。  |
| sources         | sources         | 用于指定自定义 OP 对应的源码文件   |
|*args         | *args          |   用于指定 Extension 的其他参数，支持的参数与 setuptools.Extension 一致。 |
| **kwargs      | **kwargs        |   用于指定 Extension 的其他参数，支持的参数与 setuptools.Extension 一致。 |
