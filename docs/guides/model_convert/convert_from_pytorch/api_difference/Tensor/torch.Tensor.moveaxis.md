## [参数完全一致]torch.Tensor.moveasis

### [torch.Tensor.moveaxis](https://pytorch.org/docs/stable/generated/torch.Tensor.moveaxis.html)

```python
torch.Tensor.moveaxis(source, destination) 
```

### [paddle.Tensor.movedim](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/moveaxis_cn.html)

```python
paddle.Tensor.moveaxis(source, destination,name = None) 
```

两者功能一致且参数用法一致，具体如下：

| PyTorch                            | PaddlePaddle                       | 备注                               |
|------------------------------------|------------------------------------|----------------------------------|
| <font> source </font>     | <font> source </font>    | 将被移动的轴的位置。                       |
| <font> destination </font> | <font> destination </font> | 轴被移动后的目标位置。                 |