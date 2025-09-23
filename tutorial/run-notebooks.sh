#!/bin/bash
set -x
set -e

cd "$(dirname "$0")"

ls | grep fnirs | grep ipynb | while read notebook; do 
    echo "[+] Running $notebook"
    jupyter nbconvert --to notebook --execute --inplace $notebook
done
