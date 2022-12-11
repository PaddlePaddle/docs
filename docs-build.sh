#!/bin/bash
 
SELFNAME=$0 
SHORT=f:p:w:hx:
LONG=docs-dir:,paddle-dir:,paddle-whl:,https-proxy:

OPTIND=1
 
show_help() {
  cat <<HELP_HELP_HELP
$SELFNAME -f <docs-dir> [-p <paddle-dir>]
$SELFNAME -f <docs-dir> [-w <paddle-whl>]
$SELFNAME -f <docs-dir>
Options:
    -f docs Project Dir
    -p Paddle Project Dir
    -w Paddle whl, local filename or http(s) file
    -h Show help
    -x Set https_proxy
HELP_HELP_HELP
}
 
while getopts $SHORT opt; do
    case "$opt" in
        f)
            FLUIDDOC_DIR=$OPTARG
            ;;
        p)
            PADDLE_DIR=$OPTARG
            ;;
        w)
            PADDLE_WHL=$OPTARG
            ;;
        x)
            https_proxy=$OPTARG
            ;;
        h)
            show_help
            exit 0
            ;;
    esac
done


if [ "$FLUIDDOC_DIR" = '' ] ; then
    echo "-f docs-dir must be specified."
    exit 5
fi


VERSIONSTR=${VERSIONSTR:=develop}

SPHINX_DOCKERIMAGE=registry.baidubce.com/paddleopen/fluiddoc-sphinx:20210610-py38
PADDLEDEV_DOCKERIMAGE=registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda10.1-cudnn7-gcc82


if [ "$PADDLE_WHL" = '' ] && [ "$PADDLE_DIR" = '' ] ; then
  docker run -it --rm --entrypoint=bash \
    -e VERSIONSTR=${VERSIONSTR} \
    -v ${FLUIDDOC_DIR}:/FluidDoc \
    -v ${FLUIDDOC_DIR}/output:/docs \
    -v ${HOME}/.doctrees:/var/doctrees \
    ${SPHINX_DOCKERIMAGE} \
    /root/fluiddoc-gendoc.sh
  exit 0
fi

if ! [ "$PADDLE_WHL" = '' ]; then
  if [[ ${PADDLE_WHL} = http* ]]; then
    docker run -it --rm --entrypoint=bash \
      -e VERSIONSTR=${VERSIONSTR} \
      -e https_proxy=${https_proxy} \
      -v ${FLUIDDOC_DIR}:/FluidDoc \
      -v ${FLUIDDOC_DIR}/output:/docs \
      -v ${HOME}/.doctrees:/var/doctrees \
      ${SPHINX_DOCKERIMAGE} \
      /root/fluiddoc-gendoc.sh ${PADDLE_WHL}
  else
    WHL_DIR=$(dirname $PADDLE_WHL)
    WHL_FN=$(basename $PADDLE_WHL)
    docker run -it --rm --entrypoint=bash \
      -e VERSIONSTR=${VERSIONSTR} \
      -e https_proxy=${https_proxy} \
      -v ${FLUIDDOC_DIR}:/FluidDoc \
      -v ${FLUIDDOC_DIR}/output:/docs \
      -v ${HOME}/.doctrees:/var/doctrees \
      -v ${WHL_DIR}:/whls \
      ${SPHINX_DOCKERIMAGE} \
      /root/fluiddoc-gendoc.sh /whls/${WHL_FN}
  fi
  exit 0
fi

if ! [ "$PADDLE_DIR" = '' ]; then
  docker run -it --rm \
    -e PADDLE_VERSION=${VERSIONSTR} \
    -e https_proxy=${https_proxy} \
    -v ${PADDLE_DIR}:/paddle \
    -v ${HOME}/.cache:/root/.cache \
    -v ${PADDLE_DIR}/build/python/dist:/paddle/build/python/dist \
    --workdir=/paddle/build \
    --entrypoint=bash \
    ${PADDLEDEV_DOCKERIMAGE} \
    -c 'cmake .. -DPY_VERSION=3.8 -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)'

  if ! [ "${AUTOFINDWHL}" = "off" ] ; then
    WHL_FULLFN=$(find ${PADDLE_DIR}/build/python/dist -type f -name 'paddlepaddle*cp38*linux_x86_64.whl' -printf "%T+\t%p\n" | sort -r | head -n 1| cut -f 2)
    WHL_FN=$(basename ${WHL_FULLFN:-paddle.whl})
  fi

  docker run -it --rm --entrypoint=bash \
    -e VERSIONSTR=${VERSIONSTR} \
    -e https_proxy=${https_proxy} \
    -v ${FLUIDDOC_DIR}/:/FluidDoc \
    -v ${FLUIDDOC_DIR}/output:/docs \
    -v ${HOME}/.doctrees:/var/doctrees \
    -v ${PADDLE_DIR}/build/python/dist:/whls \
    ${SPHINX_DOCKERIMAGE} \
    /root/fluiddoc-gendoc.sh /whls/${WHL_FN}
  exit 0
fi

