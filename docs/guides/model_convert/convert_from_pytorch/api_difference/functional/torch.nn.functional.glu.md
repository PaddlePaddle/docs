## [ 仅参数名不一致 ]torch.nn.functional.glu

### [torch.nn.functional.glu](https://pytorch.org/docs/stable/generated/torch.nn.functional.glu.html?highlight=glu#torch.nn.functional.glu)

```python
torch.nn.functional.glu(input, dim=- 1)
```

### [paddle.nn.functional.glu](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/glu_cn.html)

```python
paddle.nn.functional.glu(x, axis=- 1, name=None)
```

两者功能一致，仅参数名不一致，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input           | x           | 表示输入的 Tensor ，仅参数名不一致。               |
| dim           | axis           | 表示划分输入 Tensor 的维度，仅参数名不一致。               |
