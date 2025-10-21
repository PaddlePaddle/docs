## [ 仅 API 调用方式不一致 ]torch.sort
### [torch.sort](https://pytorch.org/docs/stable/generated/torch.sort.html?highlight=sort#torch.sort)
```python
torch.sort(input,
           dim=-1,
           descending=False,
           stable=False,
           *,
           out=None)
```

### [paddle.sort](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/sort_cn.html#paddle.sort)
```python
paddle.sort(x,
            axis=-1,
            descending=False,
            stable=False,
            name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，同时两个 api 的返回参数类型不同，具体如下：
