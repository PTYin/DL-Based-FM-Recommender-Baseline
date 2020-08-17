#!/bin/bash

cd src/
embedding_size=$1
datasets=("automotive" "books" "clothing" "ml-1m" "office" "ticket")
models=("DeepFM" "XDeepFM")
config_dir="../config/rating"

for dataset in ${datasets[@]} do
  for model in ${models[@]} do
    if [ ! -d "/home/share/yinxiangkun/log/libFM/${embedding_size}/${dataset}/" ]; then
      mkdir "/home/share/yinxiangkun/log/libFM/${embedding_size}/${dataset}/"
    fi
    echo "${config_dir}/${embedding_size}/${dataset}/${model}"
#    python app.py --config "${config_dir}/${embedding_size}/${dataset}/${model}" > "/home/share/yinxiangkun/log/libFM/${embedding_size}/${dataset}/${model}.txt"
  done
done

# Usage: `nohup ./auto_train.sh >> nohup.log 2>&1 &`