## [ 仅参数名不一致 ]torch.argmin
### [torch.argmin](https://pytorch.org/docs/stable/generated/torch.argmin.html?highlight=argmin#torch.argmin)

```python
torch.argmin(input,
             dim=None,
             keepdim=False)
```

### [paddle.argmin](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/argmin_cn.html#argmin)

```python
paddle.argmin(x,
              axis=None,
              keepdim=False,
              dtype='int64',
              name=None)
```

其中 Paddle 相比 PyTorch 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的多维 Tensor ，仅参数名不一致。                   |
|  dim  |  axis     | 指定进行运算的轴，仅参数名不一致。  |
| keepdim |  keepdim | 是否在输出 Tensor 中保留减小的维度。  |
| - |  dtype    | 输出 Tensor 的数据类型， PyTorch 无此参数， Paddle 保持默认即可。  |
