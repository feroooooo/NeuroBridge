#!/bin/bash
set -e
urls=(
"https://s3.amazonaws.com/openneuro.org/ds004212/derivatives/preprocessed/preprocessed_P1-epo.fif?versionId=_KO81vnVItjzcxrR91AFeHh8sA7YiuB9 data/things_meg/raw_meg/preprocessed_P1-epo.fif"
"https://s3.amazonaws.com/openneuro.org/ds004212/derivatives/preprocessed/preprocessed_P2-epo.fif?versionId=yYjGg9sMwgC3FozM8Sc_HXmKoSd_6Qnv data/things_meg/raw_meg/preprocessed_P2-epo.fif"
"https://s3.amazonaws.com/openneuro.org/ds004212/derivatives/preprocessed/preprocessed_P3-epo.fif?versionId=oaTbATaOOvX.UtsD5gA9ET7RJoZljTsB data/things_meg/raw_meg/preprocessed_P3-epo.fif"
"https://s3.amazonaws.com/openneuro.org/ds004212/derivatives/preprocessed/preprocessed_P4-epo.fif?versionId=LM9PMqdfCcx8h9qjfU_W69WU0Ol9ECdl data/things_meg/raw_meg/preprocessed_P4-epo.fif"
"https://s3.amazonaws.com/openneuro.org/ds004212/sourcedata/sample_attributes_P1.csv?versionId=w8BUbZcwHpY.6GxyOjtFSmTnc52bhBLs data/things_meg/sourcedata/sample_attributes_P1.csv"
"https://s3.amazonaws.com/openneuro.org/ds004212/sourcedata/sample_attributes_P2.csv?versionId=Evg76igx3TBV6JfoCAMLk5F8NB9O2niW data/things_meg/sourcedata/sample_attributes_P2.csv"
"https://s3.amazonaws.com/openneuro.org/ds004212/sourcedata/sample_attributes_P3.csv?versionId=P18epEYFJ3xxvxm.5B9eTXC.zZ7LP7Af data/things_meg/sourcedata/sample_attributes_P3.csv"
"https://s3.amazonaws.com/openneuro.org/ds004212/sourcedata/sample_attributes_P4.csv?versionId=eZ.cG_jUR9G4OX1K23_4.DV99kj2KUW0 data/things_meg/sourcedata/sample_attributes_P4.csv"
)

for entry in "${urls[@]}"; do
    read -r url path <<< "$entry"
    echo "Downloading：$path"
    curl -C - --retry 5 --retry-delay 5 --create-dirs -L -o "$path" "$url"
done