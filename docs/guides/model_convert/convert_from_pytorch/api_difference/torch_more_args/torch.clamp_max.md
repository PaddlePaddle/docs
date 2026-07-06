## [ torch 参数更多 ]torch.clamp_max

### [torch.clamp_max]()
```python
torch.clamp_max(input,
            max=None,
            *,
            out=None)
```

### [paddle.clamp_max](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/clamp_max_cn.html#paddle.clamp_max)
```python
paddle.clamp_max(x,
            max=None,
            *,
            out=None)
```

两者功能一致，PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input       |  x             | 表示输入的 Tensor ，仅参数名不一致。  |
| max         | max            | 表示裁剪的最大值。            |
|  out        | out            | 表示输出的 Tensor。            |
