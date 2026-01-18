## [ 仅 API 调用方式不一致 ]transformers.PreTrainedTokenizer
### [transformers.PreTrainedTokenizer](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PythonBackend)
```python
transformers.PreTrainedTokenizer(**kwargs)
```

### [paddleformers.transformers.PretrainedTokenizer](https://github.com/PaddlePaddle/PaddleFormers/blob/8a544d93fab8984c223218d2e31328ffefb01f67/paddleformers/generation/configuration_utils.py#L62)
```python
paddleformers.transformers.PretrainedTokenizer(**kwargs)
```

两者功能一致，均接受可变参数 kwargs 初始化。
