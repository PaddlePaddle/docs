# PaddlePaddle docs

[English](./README.md) | [简体中文](./README_cn.md) | 日本語


[PaddlePaddle ドキュメント](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/index_cn.html)で紹介されている内容のソースファイルです。

注: 英語版 API ドキュメントは一般的に [PaddlePaddle/Paddle](https://github.com/PaddlePaddle/Paddle) のドキュメントです。[他の PaddlePaddle プロジェクト](https://www.paddlepaddle.org.cn/overview)のドキュメントはそれぞれの方法で管理されています。

## コードベースの構造

- [docs](docs): PaddlePaddle 2.0 以上のドキュメントソースファイル。
- [docs/api](docs/api): PaddlePaddle API ドキュメント。
- [docs/guides](docs/guides): PaddlePaddle のガイドドキュメント。
- [docs/tutorial](docs/tutorial): PaddlePaddle チュートリアルのドキュメント。
- [ci_scripts](ci_scripts): ドキュメント CI スクリプト。

## ビルド方法

- 事前条件
  - docker
- 使用方法
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
  docs の出力は docs/output に生成されます。

## コントリビュート方法

詳細は [CONTRIBUTING.md](./CONTRIBUTING.md) をご覧ください。

## ライセンス



[Apache License 2.0](LICENSE)
