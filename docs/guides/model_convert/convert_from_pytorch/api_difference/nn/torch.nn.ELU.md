## [ torch 参数更多 ]torch.nn.ELU
### [torch.nn.ELU](https://pytorch.org/docs/stable/generated/torch.nn.ELU.html?highlight=elu#torch.nn.ELU)

```python
torch.nn.ELU(alpha=1.0,
             inplace=False)
```

### [paddle.nn.ELU](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/ELU_cn.html#elu)

```python
paddle.nn.ELU(alpha=1.0,
              name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| alpha           | alpha         | 表示公式中的超参数。        |
| inplace       | -            | 在不更改变量的内存地址的情况下，直接修改变量的值，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。    |
