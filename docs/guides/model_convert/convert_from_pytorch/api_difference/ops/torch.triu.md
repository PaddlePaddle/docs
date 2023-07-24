## [ torch 参数更多]torch.triu

### [torch.triu](https://pytorch.org/docs/stable/generated/torch.triu.html?highlight=triu#torch.triu)

```python
torch.triu(input,diagonal=0,*,out=None)
```

### [paddle.triu](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/triu_cn.html)

```python
paddle.triu(input,diagonal=0,name=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch | PaddlePaddle | 备注 |
| ------- | ------- | ------- |
| input | input | 表示输入的 Tensor 。 |
| diagonal | diagonal | 指定的对角线，默认值为 0 ，表示主对角线。如果 diagonal > 0 ，表示主对角线之上的对角线；如果 diagonal < 0 ，表示主对角线之下的对角线。 |
| out | - | 表示输出的 Tensor ， Paddle 没有此参数，需要进行转写。 |

### 转写示例

#### out: 输出的 Tensor

```python
# Pytorch 写法
torch.triu(input,diagonal,out=output)


# Paddle 写法
paddle.assign(paddle.triu(input,diagonal),output)
```
