## [torch 参数更多 ]torch.cross

### [torch.cross](https://pytorch.org/docs/stable/generated/torch.cross.html?highlight=cross#torch.cross)

```python
torch.cross(input,
            other,
            dim=None,
            *,
            out=None)
```

### [paddle.cross](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/cross_cn.html#cross)

```python
paddle.cross(x,
             y,
             axis=None,
             name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor ，仅参数名不一致。                          |
| other         | y            | 输入的 Tensor ，仅参数名不一致。                          |
| dim           | axis         | 沿着此维进行向量积操作，仅参数名不一致。                   |
| out           | -            | 表示输出的 Tensor ，Paddle 无此参数，需要转写。      |


### 转写示例
#### out：指定输出
```python
# PyTorch 写法
torch.cross(input, other, dim=1, out=y)

# Paddle 写法
paddle.assign(paddle.cross(input, other, axis=1), y)
```
