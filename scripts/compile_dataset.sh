#!/usr/bin/env bash

set -e 

cd OpenAmp

if ! command -v conda &> /dev/null; then
    echo "conda is required"
    exit 1
fi

eval "$(conda shell.bash hook)"

if ! [[ $(conda env list | grep "open-amp-demo") ]]; then
    echo "Making env. This might take a while"
    conda env create -f environment.yaml
fi

conda activate "open-amp-demo"

python compile_input_data.py -o '../data/Ibanez2820-DI' -i '../data/IDMT-SMT-GUITAR_V2/dataset4/Ibanez 2820'
python compile_input_data.py -o '../data/Carrer SG' -i '../data/IDMT-SMT-GUITAR_V2/dataset4/Career SG'      
