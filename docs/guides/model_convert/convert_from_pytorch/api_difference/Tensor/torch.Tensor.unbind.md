# [仅参数名不一致]torch.Tensor.unbind

[torch.Tensor.unbind](https://pytorch.org/docs/stable/generated/torch.unbind.html#torch-unbind)

```python
torch.unbind(input, dim=0)
```

[paddle.Tensor.unbind](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/unbind_cn.html#unbind)

```python
paddle.unbind(input, axis=0)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：

| PyTorch | PaddlePaddle |              备注              |
| :-----: | :----------: | :----------------------------: |
|   dim   |     axis     | 表示进行运算的轴，仅参数名不同 |
