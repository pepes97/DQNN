# NN_Project

Quaternion Neural Networks project for 3D Sound Source

## Requirements
### Create a virtual environment
```bash
python3 -m pip install virtualenv
```
```bash
virtualenv venv
virtualenv -p /usr/bin/python3 venv
```
### Activate the environment
The sequent command has to be executed before you want to run the code. It creates a virtual environment in which you can install all the packages used in your project:
```bash
source venv/bin/activate
```
**You have to activate the virtual env before you can run the code**
### Install requirements (only once)
Install packages (only the first time)
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

## Train
Once you have extracted the features you can train the network
```bash
python3 ./seld.py <job-id> <task-id>
```
<ul>
<li><b>job-id</b> : is the name used to save models, etc...</li> 
<li><b>task-id</b> : is a number referred to the configurations used (<b>999</b> for quick test)</li>
</ul>
</br>
</br>

This runs the default configs
```bash
python3 ./seld.py
```