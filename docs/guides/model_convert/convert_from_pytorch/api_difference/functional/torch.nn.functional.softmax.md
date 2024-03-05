## [ 仅参数名不一致 ]torch.nn.functional.softmax

### [torch.nn.functional.softmax](https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html?highlight=softmax#torch.nn.functional.softmax)

```python
torch.nn.functional.softmax(input,
                            dim=None,
                            dtype=None)
```

### [paddle.nn.functional.softmax](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/softmax_cn.html)

```python
paddle.nn.functional.softmax(x,
                             axis=- 1,
                             dtype=None,
                             name=None)
```

两者功能一致，仅参数名不一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input           | x           | 表示输入 Tensor ，仅参数名不一致。               |
| dim           | axis           | 表示对输入 Tensor 运算的轴 ，仅参数名不一致。               |
| dtype          | dtype           | 表示输入 Tensor 的数据类型 。               |
