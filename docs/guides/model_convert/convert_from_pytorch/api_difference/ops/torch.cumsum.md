## [torch 参数更多 ]torch.cumsum

### [torch.cumsum](https://pytorch.org/docs/stable/generated/torch.cumsum.html?highlight=cumsum#torch.cumsum)

```python
torch.cumsum(input,
             dim,
             *,
             dtype=None,
             out=None)
```

### [paddle.cumsum](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/cumsum_cn.html#cumsum)

```python
paddle.cumsum(x,
              axis=None,
              dtype=None,
              name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor ，仅参数名不同。                          |
| dim           | axis         | 指明需要累加的维度，仅参数名不同。                       |
| dtype         | dtype        | 输出 Tensor 的数据类型，默认为 None，参数名相同。         |
| out           | -            | 表示输出的 Tensor ，需要进行转写。                      |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.cumsum([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], dim=-1, dtype='float64', out=y)

# Paddle 写法
paddle.assign(paddle.cumsum([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], dim=-1, dtype='float64'), y)
```
