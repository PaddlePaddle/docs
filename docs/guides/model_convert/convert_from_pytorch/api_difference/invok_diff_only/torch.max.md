## [ 仅 API 调用方式不一致 ]torch.max
输入一个 Tensor 对应 paddle.max，输入两个 Tensor 对应 paddle.maximum，因此有两组差异分析，分别如下：

-------------------------------------------------------------------------------------------------

### [torch.max](https://pytorch.org/docs/stable/generated/torch.max.html?highlight=max#torch.max)
```python
torch.max(input,
          dim=None,
          keepdim=False,
          *,
          out=None)
```

### [paddle.max](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/max_cn.html#max)
```python
paddle.max(x,
           axis=None,
           keepdim=False,
           name=None)
```

其中 PyTorch 与 Paddle 指定 `dim` 后返回值不一致，具体如下：
