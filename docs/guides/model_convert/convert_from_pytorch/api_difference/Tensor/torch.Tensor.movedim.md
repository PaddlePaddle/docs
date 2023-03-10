## [仅参数名称不一致]torch.Tensor.movedim

### [torch.Tensor.movedim](https://pytorch.org/docs/stable/generated/torch.Tensor.movedim.html)

```python
torch.Tensor.movedim(source, destination) 
```

### [paddle.Tensor.movedim](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/moveaxis_cn.html)

```python
paddle.Tensor.moveaxis(source, destination,name = None) 
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：

| PyTorch                            | PaddlePaddle                       | 备注                               |
|------------------------------------|------------------------------------|----------------------------------|
| <font color='red'> source </font>     | <font color='red'> source </font>    | 将被移动的轴的位置。                       |
| <font color='red'> destination </font> | <font color='red'> destination </font> | 轴被移动后的目标位置。                 |