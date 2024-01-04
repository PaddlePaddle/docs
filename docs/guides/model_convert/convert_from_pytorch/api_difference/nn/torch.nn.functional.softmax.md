## [ torch 参数更多 ]torch.nn.functional.softmax

### [torch.nn.functional.softmax](https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html#torch.nn.functional.softmax)

```python
torch.nn.functional.softmax(input, dim=None, _stacklevel=3, dtype=None)
```

### [paddle.nn.functional.softmax](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/softmax_cn.html#softmax)

```python
paddle.nn.functional.softmax(x, axis=- 1, dtype=None, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch | PaddlePaddle | 备注                                                  |
|:-------:|:------------:| :---------------------------------------------------: |
| input   |   x           |  表示输入张量，仅参数名不一致。           |
| dim     | axis         |  表示对输入 Tensor 进行运算的轴，仅参数名不一致。            |
| dtype   |   dtype      |  表示返回张量所需的数据类型。  |
| _stacklevel | - | Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
