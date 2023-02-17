## torch.movedim
### [torch.movedim](https://pytorch.org/docs/stable/generated/torch.movedim.html?highlight=movedim#torch.movedim)

```python
torch.movedim(input,
              source,
              destination)
```

### [paddle.movedim](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/moveaxis_cn.html#moveaxis)

```python
paddle.moveaxis(x,
                source,
                destination,
                name=None)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 N-D Tensor。                   |
