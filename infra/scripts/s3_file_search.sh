#!/bin/bash

BUCKET_NAME=$1
PREFIX=$2

objects=$(aws s3api list-objects --bucket $BUCKET_NAME | jq -r '.Contents[].Key')

for filename in $objects; do
  filename=$(echo "$filename" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
  if [[ $filename == *"$PREFIX"* ]]; then
    echo $filename
  fi
done
