# Question based on Handling Computer Systems

Table of Contents
1. [Google Colab & Drive](#1-google-colab--drive)
2. [Kaggle Dataset, Model, Competition and Compute](#2-kaggle-dataset-model-competition--compute)
3. [Don't Know](#)





### 1. Google Colab & Drive

Quest-1.1 : How to access the files stored in google drive to colab ?
> 1. We can directely mount google drive on colab compute machine by just clicking left side folder icon then you will see on topside a symbol of drive just hit it and then your drive will get mounted.

> 2. You can also write code inside the colab notebook 
```python
from google.colab import drive
drive.mount('/content/drive')
```
> execute the cell then you drive will get mounted.

Quest-1.2 : How to make copy of the drive folder to the colab runtime storage ?
```python
# 1. mount drive
from google.colab import drive
drive.mount('/content/drive')
# 2. change directory if needed
%cd /content/drive/folderName
# 3. copy the folder using linux sys command
!cp -r /content/drive/MyDrive/FolderName /content

```
> Note : The copied folder is independent of the original folder of drive means the changes that we make locally to the colab folder will not reflect onto the drive folder vica versa.

Quest-1.3 : How to make link between drive and colab folder such that python or any other module depenedency for the our project can be installed or used easly ?
> In linux we have `!ln -s /content/drive/MyDrive/FolderName /content` used to create the symbolic link between one memory location to another of some content but in colab it does't supported therefore there is an alternate way we can use to do our task
```python
# 1. mount drive
from google.colab import drive
drive.mount('/content/drive')
# 2. change directory if needed
%cd /content/drive/folderName
# 3. importing local package directely from the drive folder
import sys
sys.path.append('/content/drive/MyDrive/FolderName')
import package

```
> Note : import that folder which has `__init__.py` file for python package.


Quest-1.4 : How to clone and make changes and update a github repository in colab and drive ?
```python

# Clone the repository
!git clone https://github.com/username/repository.git

# Navigate to the local repo
%cd /path/to/local/repo/

# Add changes
!git add .

# Commit changes
!git commit -m "Added files from Google Drive"

# Push changes to GitHub
!git push origin main

```




















### 2. Kaggle Dataset, Model, Competition & Compute


### 3. 