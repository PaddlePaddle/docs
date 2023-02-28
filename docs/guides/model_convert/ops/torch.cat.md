## torch.cat
### [torch.cat](https://pytorch.org/docs/1.13/generated/torch.cat.html?highlight=cat#torch.cat)

```python
torch.cat(input,
           *,
           out=None)
```

### [paddle.concat](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/concat_cn.html#concat)

```python
paddle.concat(x,
              name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor。                                      |
| out           | -            | 表示输出的 Tensor，PaddlePaddle 无此参数。               |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.cat([x, y], out=y)

# Paddle 写法
y = paddle.concat([x, y])
```
