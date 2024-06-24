## [ 仅参数名不一致 ]torch.all

### [torch.all](https://pytorch.org/docs/stable/generated/torch.all.html?highlight=all#torch.all)

```python
torch.all(input, dim=None, keepdim=False, *, out=None)
```

### [paddle.all](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/all_cn.html#all)

```python
paddle.all(x,
           axis=None,
           keepdim=False,
           name=None)
```

其中 Paddle 与 PyTorch 的 `input` 参数所支持的类型不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input        | x           | 输入的多维 Tensor ，参数完全一致。                   |
| dim    |  axis     | 表示运算的维度，仅参数名不一致。        |
| keepdim    |  keepdim  | 是否在输出 Tensor 中保留减小的维度，参数完全一致。  |
