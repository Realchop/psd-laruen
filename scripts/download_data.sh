#!/usr/bin/env bash

set -e

mkdir -p data
cd data

curl -LO 'https://github.com/GuitarML/ToneLibrary/releases/download/v1.0/Proteus_Tone_Packs.zip'
unzip Proteus_Tone_Packs.zip
rm Proteus_Tone_Packs.zip

URL='https://zenodo.org/records/7544110/files/IDMT-SMT-GUITAR_V2.zip'

if command -v aria2c &> /dev/null; then
    aria2c -x 16 -s 16 $URL
else
    echo "aria2c not found. Using curl (aria2c would be faster). Try installing aria2, it will be quicker"
    curl -LO $URL
fi
unzip IDMT-SMT-GUITAR_V2.zip
rm IDMT-SMT-GUITAR_V2.zip

cd ..

