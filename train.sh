#!/bin/bash

cd src/
embedding_size=$1
config_dir="../config/rating"

for dataset in "automotive" "books" "clothing" "ml-1m" "office" "ticket" do
  for model in "DeepFM" "XDeepFM" do
    if [ ! -d "/home/share/yinxiangkun/log/libFM/${embedding_size}/${dataset}/" ]; then
      mkdir "/home/share/yinxiangkun/log/libFM/${embedding_size}/${dataset}/"
    fi
    echo "${config_dir}/${embedding_size}/${dataset}/${model}"
#    python app.py --config "${config_dir}/${embedding_size}/${dataset}/${model}" > "/home/share/yinxiangkun/log/libFM/${embedding_size}/${dataset}/${model}.txt"
  done
done

# Usage: `nohup ./auto_train.sh >> nohup.log 2>&1 &`