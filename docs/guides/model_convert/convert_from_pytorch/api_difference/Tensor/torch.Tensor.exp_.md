## [ 仅参数名不一致 ]torch.Tensor.exp\_

### [torch.exp\_](https://pytorch.org/docs/stable/generated/torch.Tensor.exp_.html?highlight=exp_#torch.Tensor.exp_)

```python
Tensor.exp_(input)
```

### [paddle.exp\_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/exp__cn.html#exp)

```python
paddle.exp_(x, name=None)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：

### 参数映射

| PyTorch                  | PaddlePaddle         | 备注                                                            |
| ------------------------ | -------------------- | --------------------------------------------------------------- |
| <center> input </center> | <center> x </center> | Inplace 版本的 exp API，对输入采用 Inplace 策略，仅参数名不同。 |
