## [参数不一致]torch.cuda.manual_seed

### [torch.cuda.manual_seed](https://pytorch.org/docs/stable/generated/torch.cuda.manual_seed.html#torch.cuda.manual_seed)

```python
torch.cuda.manual_seed(seed)
```

### [paddle.seed](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/seed_cn.html)

```python
paddle.seed(seed)
```

功能一致，返回类型不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                 |
|---------|--------------|----------------------------------------------------|
| seed    | seed         | 表示设置的的随机种子(int)。                                   |
| -       | 返回值          | PyTorch 无返回值，Paddle 返回 Generator(全局默认 generator 对象)。 |

### 转写示例
#### 返回值
```python
# torch 写法
torch.cuda.manual_seed(100)

# paddle 写法
gen = paddle.seed(100)
```
