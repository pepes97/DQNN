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
Download the dataset. This script downloads the TAU-Dataset and unzips the folders.
```bash
python3 download_dataset.py
```

The sequent script prepare the folders as the program expects to receive them. It renames all the entries of the dataset according to the development set and evaluation set, and divides them according to the number of overlapping (ov1 and ov2). It creates two main folders with the same content (**TAU Dataset** and **TAU Dataset Seld**), these will be use later to extract the features for the two different nets.
```bash
python3 prepare_dataset.py
```

Now you should have two folder inside Dataset folder: **TAU Dataset** and **TAU Dataset Seld**.
Inside each folder there are two folders for wav and labels.

## Features extraction
```bash
cd Code
```
```bash
python3 batch_feature_extraction.py
python3 batch_feature_extraction_seld.py
```
The first script extracts features for the quaternion net.
The second script extracts features for the traditional net.
These add the extracted features to the correct folder.

## Train
### Quaternion Net
Once you have extracted the features you can train the network
```bash
python3 ./seld.py <name> --author <author> --params <params_num>
```
<ul>
<li><b>name</b> : is the name used to save the model and other info. If the name already exists it continues the training from the saved epoch, with saved info (metrics, etc..).
<li><b>author</b> : is the name of the person who runs the training</li> 
<li><b>params_num</b> : is a number referred to the configurations used (<b>999</b> for quick test, <b>10</b> for ov1, <b>20</b> for ov2). It is possible to implement other configurations (TO DO).</li>
</ul>

### Traditional Convolutional Net 
```bash
python3 ./seld_cnn.py <name> --author <author> --params <params_num>
```

<ul>
<li><b>params_num</b> : is a number referred to the configurations used (<b>11</b> for ov1, <b>21</b> for ov2). It is possible to implement other configurations (TO DO).</li>
<li>other params as before
</ul>

## Tensorboard
To visualize the plots we use Tensorboard
```bash
tensorboard --logdir ./logs
```
```bash
tensorboard --logdir ./Code/logs
```
<b>logdir</b> : logdir depends on where you are

### Tensorboard on Colab
This loads the extension in Colab
```bash
%load_ext tensorboard
```
This opens the window of tensorboard (inside Colab you have to run this <b>before</b> run the training, in this way you can see the progresses in real time)
```bash 
%tensorboard --logdir ./logs
```
