#!/bin/sh
ls /models
# shellcheck disable=SC1009
/llama.cpp/build/bin/llama-server -c 32000 -m /models/model.gguf \
--host 0.0.0.0 \
 --port 8080 \
 --gpu-layers 35 \
 -b 512 \
 -ub 256 \
 --mmap \
   --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --temp 0.1