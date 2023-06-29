## [ torch 参数更多 ]torch.cummin

### [torch.cummin](https://pytorch.org/docs/stable/generated/torch.cummin.html)

```python
torch.cummin(input,
          dim,
          *,
          out=None)
```

### [paddle.cummin](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/cummin_cn.html)

```python
paddle.cummin(x,
            axis=None,
            dtype='int64',
            name=None)
```

两者功能一致，torch 参数更多，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input          | x          | 表示输入的 Tensor,仅参数名不一致。                        |
| dim          | axis         | 用于指定 index 获取输入的维度, PyTorch 该参数没有默认值, Paddle 当该参数为 None 时，会将输入展开为一维变量再进行累计最小值计算。             |
| -        | dtype |  指定输出索引的数据格式, PyTorch 无此参数, Paddle 保持默认即可。 |
| out        | - |  表示输出的 Tensor, Paddle 无此参数，需要进行转写。 |

### 转写示例
#### out：指定输出

```python
# Pytorch 写法
torch.cummin(x,1, out=(values, indices))

# Paddle 写法
paddle.assign(paddle.cummin(x,y), (values, indices))
```
