#!/bin/bash

rm ./imgs/attn_heatmaps/*
json=`sed "${2}q;d" $1`
img="`dirname $1`/`echo $json | jq -r '.img'`"
crop=`echo $json | jq -r '.box'`

echo "img: $img"
echo "crop: $crop"
echo "label: `echo $json | jq -r '.tag'`"

/usr/bin/python3 visualize_attn.py ./saved_models/hc_full.ckpt $img --crop "$crop" --resize
