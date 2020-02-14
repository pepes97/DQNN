# NN_Project

Quaternion Neural Networks project for 3D Sound Source

## Requirements
```bash
python3 -m pip install -r ./Code/requirements.txt
```

## Dataset
Unzip archives inside Dataset folder.
You should have two folders (foa_dev and metadata_dev).
```bash
python3 ./prepare_dataset.py
```

Now you should have one folder inside Dataset folder: TAU Dataset.
Inside TAU Dataset there are two folders for wav and labels.

## Features extraction
```bash
cd ./Code
```
```bash
python3 ./batch_feature_extraction.py
```
This adds the extracted features to the TAU Dataset folder.