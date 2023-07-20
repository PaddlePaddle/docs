## [ 仅参数名不一致 ]torch.utils.dlpack.from_dlpack
### [torch.utils.dlpack.from_dlpack](https://pytorch.org/docs/stable/dlpack.html?highlight=torch+utils+dlpack+from_dlpack#torch.utils.dlpack.from_dlpack)

```python
torch.utils.dlpack.from_dlpack(ext_tensor)
```

### [paddle.utils.dlpack.from_dlpack](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/utils/dlpack/from_dlpack_cn.html)

```python
paddle.utils.dlpack.from_dlpack(dlpack)
```

两者功能一致，参数名不一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| ext_tensor        | dlpack        | DLPack，即带有 dltensor 的 PyCapsule 对象，仅参数名不一致。   |
