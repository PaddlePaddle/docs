## torch.argmin
### [torch.argmin](https://pytorch.org/docs/stable/generated/torch.argmin.html?highlight=argmin#torch.argmin)

```python
torch.argmin(input,
            dim=None,
            keepdim=False)
```

### [paddle.argmin](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/argmin_cn.html#argmin)

```python
paddle.argmin(x,
            axis=None,
            keepdim=False,
            dtype='int64',
            name=None)
```
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input        | x            | 输入的多维 Tensor。                   |
| dim          | axis         | 指定对输入 Tensor 进行运算的轴。 |
