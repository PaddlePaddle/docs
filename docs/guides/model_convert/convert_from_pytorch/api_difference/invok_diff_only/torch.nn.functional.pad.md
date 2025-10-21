## [ 仅 API 调用方式不一致 ]torch.nn.functional.pad
### [torch.nn.functional.pad](https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html)
```python
torch.nn.functional.pad(input,
                            pad,
                            mode='constant',
                            value=None)
```

### [paddle.nn.functional.pad](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/pad_cn.html#pad)
```python
paddle.nn.functional.pad(x,
                            pad,
                            mode='constant',
                            value=0.0,
                            data_format=None,
                            pad_from_left_axis=True,
                            name=None)
```

两者功能一致，仅调用方式不一致。
