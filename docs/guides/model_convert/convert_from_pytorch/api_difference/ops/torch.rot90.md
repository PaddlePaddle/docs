## [仅参数名不一致]torch.rot90

### [torch.rot90](https://pytorch.org/docs/stable/generated/torch.rot90.html?highlight=torch+rot90#torch.rot90)
```python
torch.rot90(input,
            k=1,
            dims=[0, 1])
```

### [paddle.rot90](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/rot90_cn.html#rot90)

```python
paddle.rot90(x,
             k=1,
             axes=[0, 1],
             name=None)
```

两者功能一致且参数用法一致，仅参数名不一致，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor ，仅参数名不一致。                   |
| k         | k            | 表示旋转方向和次数。                   |
| dims          | axes         | axes 指定旋转的平面，维度必须为 2 ，仅参数名不一致。   |
