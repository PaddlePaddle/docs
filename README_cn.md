# PaddlePaddle docs

[English](./README.md) | 简体中文 | [日本語](./README_ja.md)

docs 是 [PaddlePaddle 官网文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/index_cn.html) 的源文件。

注意：英文版 API 文档直接从[PaddlePaddle/Paddle](https://github.com/PaddlePaddle/Paddle) 的 docstring 中生成，[飞桨其他项目](https://www.paddlepaddle.org.cn/overview)的文档分别在其对应的位置中管理。

## 仓库结构

- [docs](docs): 飞桨框架 2.0 以及之后版本文档的源文件。
- [docs/api](docs/api): 飞桨中文 API 文档的源文件。
- [docs/guides](docs/guides): 飞桨官方教程的源文件。
- [docs/tutorial](docs/tutorial): 飞桨相关案例的源文件。
- [ci_scripts](ci_scripts): docs CI 相关的文件。

## 如何构建文档

- 依赖
  - docker
- 构建过程
  - step1: clone docs
    ```
    git clone https://github.com/PaddlePaddle/docs
    ```
  - step2: build docs
    ```
    cd docs
    mkdir output
    bash docs-build.sh -f absolute_path_docs
    ```
  - step3: preview docs
  输出的文件会保存在 docs/output 路径下。

## 贡献

我们非常欢迎你贡献文档！你可以参考 [贡献指南](CONTRIBUTING_cn.md) 直接参与。

## License

[Apache License 2.0](LICENSE)

# Logger

new doc line
