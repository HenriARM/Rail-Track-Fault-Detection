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

# Deployment (on Linux)
Used [example](https://github.com/pytorch/serve/tree/master/examples/image_classifier/densenet_161)

1. Install TorchServe Docker image (docker should be installed)
    ```.bash
    $ docker pull pytorch/torchserve # 4bee499b0373, image requires 4GB

    ```
2. Make sure that Java 11 is installed. [Guide](https://sysadminxpert.com/how-to-upgrade-java-8-to-java-11-on-ubuntu-20/)
    ```.bash
    $ sudo apt install openjdk-11-jdk
    $ java --version
    ```
    Output:
    ```
    openjdk version "11.0.17" 2022-10-18
    OpenJDK Runtime Environment (build 11.0.17+8-post-Ubuntu-1ubuntu220.04)
    OpenJDK 64-Bit Server VM (build 11.0.17+8-post-Ubuntu-1ubuntu220.04, mixed mode, sharing)
    ```

3. Convert saved model to traced using `convert_to_traced_module.py` (edit path of downloaded model `.pt` from Drive)
4. Create a Model Archive (`.mar` file):

    Install requirements in conda for all next steps
    ``` .bash
    $ pip install -r deployment_requirements.txt
    ```

    Run `torch-model-archiver`
    ```.bash
    $ torch-model-archiver --model-name model --version 1.0  --serialized-file model.pt --handler custom_handler.py
    ```

    Move `.mar` file to separate folder
    ```.bash
    $ mkdir model_store
    $ mv model.mar model_store/
    ```

    For more information check [docs](https://github.com/pytorch/serve/tree/master/model-archiver#creating-a-model-archive)


5. Deploy TorchServe:

    ```.bash
    $ torchserve --start --ncs --model-store model_store --models model=model.mar
    ```

6. Check TorchServe status and Test models:

    Model is deployed by default on localhost in the ports 8080 (Inference API), 8081 (Management API) and 8082(Metrics API)

    Ping:

    ```.bash
    $ curl http://localhost:8080/ping
    ```
    If everything works, it should output:
    ```
    {
        "status": "Healthy"
    }
    ```

    List deployed model info:
    ```.bash
    $ curl http://localhost:8081/models/model
    ```

    Inference test image
    ```.bash
    $ curl -X POST http://localhost:8080/predictions/model -T test_defective.jpg
    ```
    Result: (class 0 - `defective`, class 1 - `non-defective`)
    ```
    {
        "class": 0
    }
    ```

7. Stop TorchServe

    ```.bash
    $ torchserve --stop
    ```

#TODO: torchserve as a docker
#TODO: get metrics

# Project structure:
* `train.py` - train Backbone for Binary Classification
* `eval.py` - run on test data saved model