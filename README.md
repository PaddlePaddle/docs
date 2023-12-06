# PaddlePaddle docs

English | [简体中文](./README_cn.md) | [日本語](./README_ja.md)


Source files for contents presented at [PaddlePaddle documentation](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/index_cn.html).

Note: English version API docs are generaly docstrings in [PaddlePaddle/Paddle](https://github.com/PaddlePaddle/Paddle), documents for [other PaddlePaddle projects](https://www.paddlepaddle.org.cn/overview) are being managed in their respective ways.

## Codebase structure

- [docs](docs): PaddlePaddle 2.0 & above docs source file.
- [docs/api](docs/api): PaddlePaddle API docs.
- [docs/guides](docs/guides): PaddlePaddle guides docs.
- [docs/tutorial](docs/tutorial): PaddlePaddle tutorial docs.
- [ci_scripts](ci_scripts): docs CI scripts.

## How to build

- pre-requirements
  - docker
- Instructions
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
  The output of docs will be generated in docs/output.

## How to contribute

PaddlePaddle welcomes documentation contributions, please see [CONTRIBUTING.md](./CONTRIBUTING.md) for details.

## License



[Apache License 2.0](LICENSE)
