#!/bin/bash
set -x
set -e

cd "$(dirname "$0")"

ls fnirs_*.ipynb | while read notebook; do 
    echo "[+] Running $notebook"
    jupyter nbconvert --to notebook --execute --inplace $notebook
done
