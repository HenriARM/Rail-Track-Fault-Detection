# Rail Track Fault Detection
Binary classification

# Setup:
Install conda and create new environment:
```.bash
$ conda create --name raildetect python 3.9
$ conda activate raildetect
(raildetect) $ pip install -r requirements.txt
```
the other way to do this:
```.bash
$ conda create --name raildetect --file conda_env.txt
```
(generated from `conda list --export > conda_env.txt`)

Install [dataset](https://www.kaggle.com/datasets/gpiosenka/railway-track-fault-detection-resized-224-x-224) and move data to the root directory 

Data structure should look like this
```
-- train
    -- Defective
    -- Non defective
-- test
    -- Defective
    -- Non defective
-- valid
    -- Defective
    -- Non defective
```


# Project structure:
* `train.py` - train Backbone for Binary Classification
* `main.py` - run on test data saved model