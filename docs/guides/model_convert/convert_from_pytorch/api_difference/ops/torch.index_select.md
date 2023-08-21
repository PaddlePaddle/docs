## [torch 参数更多 ]torch.index_select
### [torch.index_select](https://www.paddlepaddle.org.cn/documentation/docs/stable/develop/api/paddle/index_select_cn.html#index-select)

```python
torch.index_select(input,
                   dim,
                   index,
                   *,
                   out=None)
```

### [paddle.index_select](https://www.paddlepaddle.org.cn/documentation/docs/stable/develop/api/paddle/index_select_cn.html#index-select)

```python
paddle.index_select(x,
                    index,
                    axis=0,
                    name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font> | <font color='red'> x </font> | 表示输入的 Tensor ，仅参数名不一致。  |
| <font color='red'> dim </font> | <font color='red'> axis </font> | 表示进行运算的轴，仅参数名不一致。  |
| <font color='red'> out </font> | -  | 表示输出的 Tensor ， Paddle 无此参数，需要转写。    |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.index_select(x, dim=1, index=index, out=y)

# Paddle 写法
paddle.assign(paddle.index_select(x, axis=1, index=index), y)
```
