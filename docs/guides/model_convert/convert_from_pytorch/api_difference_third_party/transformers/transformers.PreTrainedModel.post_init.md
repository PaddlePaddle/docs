## [组合替代实现]transformers.PreTrainedModel.post_init

### [transformers.PreTrainedModel.post_init](https://hf-mirror.com/docs/transformers/v4.42.0/en/main_classes/model#transformers.PreTrainedModel.post_init)

```python
transformers.PreTrainedModel.post_init()
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# Pytorch 写法
model.post_init()

# Paddle 写法
if hasattr(model, 'init_weights'):
    model.init_weights()
elif hasattr(model, '_init_weights'):
    model._init_weights()
```
