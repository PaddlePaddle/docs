## [ torch 参数更多 ]torch.nn.functional.mish
### [torch.nn.functional.mish](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.mish.html#torch.nn.functional.mish)
```python
torch.nn.functional.mish(input,
                         inplace=False)
```

### [paddle.nn.functional.mish](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/mish_cn.html#paddle.nn.functional.mish)
```python
paddle.nn.functional.mish(x,
                         name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input          | x         | 表示输入的 Tensor ，仅参数名不一致。                                     |
| inplace     | -            | 表示在不更改变量的内存地址的情况下，直接修改变量的值，主要功能为节省显存，一般对网络训练影响不大，Paddle 无此参数，可直接删除。 |
