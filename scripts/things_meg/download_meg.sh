#!/bin/bash
set -e
urls=(
"https://s3.amazonaws.com/openneuro.org/ds004212/derivatives/preprocessed/preprocessed_P1-epo-1.fif?versionId=VEFpBYggusLPxNIKlN21eEtrjpo4DcBP data/things_meg/raw_meg/preprocessed_P1-epo-1.fif"
"https://s3.amazonaws.com/openneuro.org/ds004212/derivatives/preprocessed/preprocessed_P1-epo-2.fif?versionId=hVx8E4e0xIymnQ3nHKFnNAhQHsr7jKlQ data/things_meg/raw_meg/preprocessed_P1-epo-2.fif"
"https://s3.amazonaws.com/openneuro.org/ds004212/derivatives/preprocessed/preprocessed_P1-epo-3.fif?versionId=1jVDhw2KOJz7b_6ApO2a9J4vy_8jqqhn data/things_meg/raw_meg/preprocessed_P1-epo-3.fif"
"https://s3.amazonaws.com/openneuro.org/ds004212/derivatives/preprocessed/preprocessed_P1-epo.fif?versionId=_KO81vnVItjzcxrR91AFeHh8sA7YiuB9 data/things_meg/raw_meg/preprocessed_P1-epo.fif"
"https://s3.amazonaws.com/openneuro.org/ds004212/derivatives/preprocessed/preprocessed_P2-epo-1.fif?versionId=yIZzH6fajlVFCen182u0nGLl5sbWKs0w data/things_meg/raw_meg/preprocessed_P2-epo-1.fif"
"https://s3.amazonaws.com/openneuro.org/ds004212/derivatives/preprocessed/preprocessed_P2-epo-2.fif?versionId=qQ9bWzptOpO8CZlBhT4c5erDOA5Do67a data/things_meg/raw_meg/preprocessed_P2-epo-2.fif"
"https://s3.amazonaws.com/openneuro.org/ds004212/derivatives/preprocessed/preprocessed_P2-epo-3.fif?versionId=uRES5t7k47rQTCDUNlFN5_rRmjNOemtW data/things_meg/raw_meg/preprocessed_P2-epo-3.fif"
"https://s3.amazonaws.com/openneuro.org/ds004212/derivatives/preprocessed/preprocessed_P2-epo.fif?versionId=yYjGg9sMwgC3FozM8Sc_HXmKoSd_6Qnv data/things_meg/raw_meg/preprocessed_P2-epo.fif"
"https://s3.amazonaws.com/openneuro.org/ds004212/derivatives/preprocessed/preprocessed_P3-epo-1.fif?versionId=JYdYRMFUzwsz8354Fz9sQQbnQ.Ozstbb data/things_meg/raw_meg/preprocessed_P3-epo-1.fif"
"https://s3.amazonaws.com/openneuro.org/ds004212/derivatives/preprocessed/preprocessed_P3-epo-2.fif?versionId=mUABa6OLpBG1Vu0A9IsFpOkgWjMBTuG8 data/things_meg/raw_meg/preprocessed_P3-epo-2.fif"
"https://s3.amazonaws.com/openneuro.org/ds004212/derivatives/preprocessed/preprocessed_P3-epo-3.fif?versionId=LadUGeGvC.s2wAqYEMXEzD4_VjD1aRJg data/things_meg/raw_meg/preprocessed_P3-epo-3.fif"
"https://s3.amazonaws.com/openneuro.org/ds004212/derivatives/preprocessed/preprocessed_P3-epo.fif?versionId=oaTbATaOOvX.UtsD5gA9ET7RJoZljTsB data/things_meg/raw_meg/preprocessed_P3-epo.fif"
"https://s3.amazonaws.com/openneuro.org/ds004212/derivatives/preprocessed/preprocessed_P4-epo-1.fif?versionId=LGG8UJiqW83Z5Mgpiv5LQDgUgs0kYn9y data/things_meg/raw_meg/preprocessed_P4-epo-1.fif"
"https://s3.amazonaws.com/openneuro.org/ds004212/derivatives/preprocessed/preprocessed_P4-epo-2.fif?versionId=tdx2wKa7QOpMjBzsN4py4Y.rYhW9vp26 data/things_meg/raw_meg/preprocessed_P4-epo-2.fif"
"https://s3.amazonaws.com/openneuro.org/ds004212/derivatives/preprocessed/preprocessed_P4-epo-3.fif?versionId=gajLhEHnXBlMs.EMnyXyFtJ_SVvw_gOT data/things_meg/raw_meg/preprocessed_P4-epo-3.fif"
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