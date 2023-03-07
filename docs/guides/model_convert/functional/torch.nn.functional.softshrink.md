## torch.nn.functional.softshrink

### [torch.nn.functional.softshrink](https://pytorch.org/docs/stable/generated/torch.nn.functional.softshrink.html?highlight=softshrink#torch.nn.functional.softshrink)

```python
torch.nn.functional.softshrink(input,
                               lambd=0.5)
```

### [paddle.nn.functional.softshrink](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/softshrink_cn.html)

```python
paddle.nn.functional.softshrink(x,
                             threshold=0.5,
                             name=None)
```

两者功能一致，仅参数名不一致，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input           | x           | 表示输入 Tensor 。               |
| lambd           | threshold           | 表示 Softshrink 公式的 threshold 值。               |
