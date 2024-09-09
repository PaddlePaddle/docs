#! /bin/bash
CURDIR=$(pwd)


script_dir=$(dirname "${BASH_SOURCE[0]}")
chmod +x $script_dir/../paddle_log
$script_dir/../paddle_log

FLUIDDOCDIR=${FLUIDDOCDIR:=/FluidDoc}
OUTPUTDIR=${OUTPUTDIR:=/docs}
CONFIGDIR=${CONFIGDIR:=${FLUIDDOCDIR}/ci_scripts/doc-build-config}
VERSIONSTR=${VERSIONSTR:=develop}
OUTPUTFORMAT=${OUTPUTFORMAT:=html}

DOCROOT=${FLUIDDOCDIR}/docs/
APIROOT=${DOCROOT}/api/

export DOCROOT


# install paddle if not installed yet.
# PADDLE_WHL is defined in ci_start.sh
pip3 list --disable-pip-version-check | grep paddlepaddle > /dev/null
if [ $? -ne 0 ] ; then
  pip3 install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple ${PADDLE_WHL}
fi


cd ${APIROOT}
python ./gen_doc.py
if [ -f ./copy_codes_from_en_doc.py ] && [ -f ./api_info_all.json ] ; then
  python ./copy_codes_from_en_doc.py --api-info ./api_info_all.json ${DOCROOT}
fi

if [ -f ${FLUIDDOCDIR}/ci_scripts/hooks/pre-doc-compile.sh ] ; then
  ${FLUIDDOCDIR}/ci_scripts/hooks/pre-doc-compile.sh
  if [ $? -ne 0 ]; then
    echo "pre-doc-compile.sh failed."
    exit 1
  fi
fi

thread=2
tmp_fifofile=/tmp/$$.fifo       # 脚本运行的当前进程ID号作为文件名
mkfifo $tmp_fifofile            # 新建一个随机fifo管道文件
exec 6<>$tmp_fifofile           # 定义文件描述符6指向这个fifo管道文件
rm $tmp_fifofile                # 清空管道内容

# for循环 往 fifo管道文件中写入$thread个空行
for ((i=0;i<$thread;i++));do
  echo
done >&6


sphinx_thread=12
for lang in en zh ; do
  read -u6
  {
    mkdir -p ${OUTPUTDIR}/${lang}/${VERSIONSTR}
    /usr/local/bin/sphinx-build -b ${OUTPUTFORMAT} -j ${sphinx_thread} -d /var/doctrees -c ${CONFIGDIR}/${lang} ${DOCROOT} ${OUTPUTDIR}/${lang}/${VERSIONSTR}
    if [ "${OUTPUTFORMAT}" = "html" ] ; then
      INDEXFILE="${OUTPUTDIR}/${lang}/${VERSIONSTR}/index_${lang}.html"
      if [ "${lang}" = "zh" ] ; then
        INDEXFILE="${OUTPUTDIR}/${lang}/${VERSIONSTR}/index_cn.html"
      fi
      if [ ! -f ${INDEXFILE} ] ; then
        /usr/local/bin/sphinx-build -b ${OUTPUTFORMAT} -j ${sphinx_thread} -d /var/doctrees -c ${CONFIGDIR}/${lang} ${DOCROOT} ${OUTPUTDIR}/${lang}/${VERSIONSTR}
      fi

      if [ "${lang}" = "en" ] ; then
        if [ -f /root/post_filter_htmls.py ] && [ ! -f ${FLUIDDOCDIR}/ci_scripts/hooks/post-doc-compile.sh ] ; then
          python /root/post_filter_htmls.py ${OUTPUTDIR}/${lang}/${VERSIONSTR}/api/
        fi
      fi
    fi
    echo >&6
  } &
done

wait            # 等到后台的进程都执行完毕
exec 6>&-       # 删除文件描述符6

if [ -f ${FLUIDDOCDIR}/ci_scripts/hooks/post-doc-compile.sh ] ; then
  ${FLUIDDOCDIR}/ci_scripts/hooks/post-doc-compile.sh ${OUTPUTDIR} ${VERSIONSTR}
fi

if [ "${VERSIONSTR:0:2}" = "1." ] ; then
  echo Done
  exit 0
fi

mkdir -p ${OUTPUTDIR}/en/${VERSIONSTR}/gen_doc_output
for f in alias_api_mapping api_label display_doc_list not_display_doc_list api_info_dict.json api_info_all.json ; do
  if [ -f ${APIROOT}/${f} ] ; then
    cp ${APIROOT}/${f} ${OUTPUTDIR}/en/${VERSIONSTR}/gen_doc_output
  fi
done
# TODO: upload OUTPUTDIR to bos

echo Done
cd ${CURDIR}
exit 0
