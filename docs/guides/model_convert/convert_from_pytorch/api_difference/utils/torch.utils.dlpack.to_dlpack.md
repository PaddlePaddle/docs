## [ 输入参数用法不一致 ]torch.utils.dlpack.to_dlpack
### [torch.utils.dlpack.to_dlpack](https://pytorch.org/docs/stable/dlpack.html?highlight=torch+utils+dlpack+to_dlpack#torch.utils.dlpack.to_dlpack)

```python
torch.utils.dlpack.to_dlpack(tensor)
```

### [paddle.utils.dlpack.to_dlpack](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/utils/dlpack/to_dlpack_cn.html)

```python
paddle.utils.dlpack.to_dlpack(x)
```

两者功能一致，参数名不一致，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| tensor        | x        | Paddle Tensor，PyTorch不支持使用关键字传参。   |
