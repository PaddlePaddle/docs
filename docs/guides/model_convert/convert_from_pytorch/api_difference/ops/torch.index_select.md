## torch.index_select
### [torch.index_select](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/index_select_cn.html#index-select)

```python
torch.index_select(input,
                   dim,
                   index,
                   *,
                   out=None)
```

### [paddle.index_select](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/index_select_cn.html#index-select)

```python
paddle.index_select(x,
                    index,
                    axis=0,
                    name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor。                                      |
| dim           | axis         | 索引轴。                                      |
| out           | -            | 表示输出的 Tensor，PaddlePaddle 无此参数。               |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.index_select(x, dim=1, index=index, out=y)

# Paddle 写法
y = paddle.index_select(x, index=index, axis=1)
```
