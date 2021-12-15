#飞桨框架IPU版推理示例

## word2vec 在docker内下载并运行示例：

例子下载命令：

```
wget https://raw.githubusercontent.com/graphcore/Paddle/develop_ipu/word2vec.cpp
```

运行命令：

```
g++ -std=c++11 word2vec.cpp -lpopart -DONNX_NAMESPACE=onnx -o word2vec
./word2vec
```
