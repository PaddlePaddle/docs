## [ 仅 API 调用方式不一致 ]torch.nn.functional.softmax
### [torch.nn.functional.softmax](https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html#torch.nn.functional.softmax)
```python
torch.nn.functional.softmax(input, dim=None, _stacklevel=3, dtype=None)
```

### [paddle.nn.functional.softmax](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/softmax_cn.html#softmax)
```python
paddle.nn.functional.softmax(x, axis=-1, dtype=None, name=None)
```

两者功能一致，仅 API 调用方式不同。
