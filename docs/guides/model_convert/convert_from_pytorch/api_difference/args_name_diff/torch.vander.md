## [ 仅参数名不一致 ]torch.vander

### [torch.vander](https://pytorch.org/docs/stable/generated/torch.vander.html?highlight=vander#torch.vander)

```python
torch.vander(x,
          N,
          increasing)
```

### [paddle.vander](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/vander_cn.html#vander)

```python
paddle.vander(x,
          n,
          increasing)
```

两者功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| x          | x          | 表示输入的 Tensor。                      |
| N          | n        | 用于指定输出的列数, 仅参数名大小写的区别。             |
| increasing        | increasing |  指定输出列的幂次顺序。如果为 True，则幂次从左到右增加。 |
