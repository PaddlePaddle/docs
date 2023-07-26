## [ 仅参数名不一致 ]torch.from_dlpack

### [torch.from_dlpack](https://pytorch.org/docs/2.0/generated/torch.from_dlpack.html?highlight=from_dlpack#torch.from_dlpack)

```python
torch.from_dlpack(ext_tensor)
```

### [paddle.utils.dlpack.from_dlpack](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/utils/dlpack/from_dlpack_cn.html)

```python
paddle.utils.dlpack.from_dlpack(dlpack)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| ext_tensor |  dlpack  | 表示输入的带有 dltensor 的 PyCapsule 对象，仅参数名不一致。  |
