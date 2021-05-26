ARG BASEIMAGE=registry.baidubce.com/paddleopen/paddle:build-gpu-cuda10.2-cudnn8-devel-ubuntu18.04-gcc8
FROM ${BASEIMAGE}

RUN git config --global user.name "PaddleCI" \
 && git config --global user.email "paddle_ci@example.com"

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10 \
 && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 10

RUN apt-get update \
 && apt-get install -yq --no-install-recommends \
    wget \
    curl \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# ci's requirements
RUN echo 'docutils\n\
scipy\n\
prettytable\n\
opencv-python==4.2.0.32\n\
markdown\n\
pre-commit\n\
pylint\n\
pytest' >requirements && \
    pip3 install --no-cache-dir -i https://mirror.baidu.com/pypi/simple -r requirements && \
    rm requirements

# paddle's requirements
RUN echo 'Pillow\n\
numpy\n\
gast==0.3.3\n\
six\n\
protobuf\n\
urllib3\n\
certifi\n\
chardet\n\
requests\n\
decorator\n\
astor' >paddle.requirements && \
    pip3 install --no-cache-dir -i https://mirror.baidu.com/pypi/simple -r paddle.requirements && \
    rm paddle.requirements
