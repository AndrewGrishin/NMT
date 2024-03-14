# Neural Machine Translation (NMT)

## Abstract

In todays world the idea of AI application spreads tremendously fast. So, nowadays there is almost no sphere of human's life which AI is not incorporated with. The main goal of this project is to provide sustainable translation system for the daily usage, allowing people of different language cultures to interact with each other. 

## Detailed scheme

The key difference of this project from other translation systems is the exploitation of the neural approach. It means that the essence of the project is the *model* used and implemented throughout the experiment.

The choice was conducted to the side of transformers. To be more specific: simple [transformer](https://arxiv.org/abs/1706.03762) model. As this paper provides researchers with the most important idea the whole world of Natural Language Processing sphere.

The project will be assembled as a CLI (Command Line Interface) programme allowing user to `train` and to `infer` the model. It leads to the idea that there will be only 2 useful files `train.py` and `infer.py`. First of all the `train.py` module should be executed give the path to train dataset. After this the weights of the trained model will be saved locally for further reuse. After that, the `infer.py` file give path to test dataset can be used.

To be even more precise let's derive the step by step scheme:

1. `python -m commands train path/to/train/dataset` $\Rightarrow$ trains the model and saves weights to the `.ptr.tar` file.
2. `python -m commands infer path/to/test/dataset` $\Rightarrow$ evaluates the model on the given dataset. In our case this step produces the `output.txt` file with translated data.

So, there are only 2 options in the programme: to train and to evaluate the model. In addition: it is important to specify the format of train and test datasets. The train file should look like:

| Target   | Source   |
| :------: | :------: |
|Sentence on the target language | Sentence in the source language|

Here **Target** is the reference to the language we are going to translate in. The **Source** is the language we are using as the input one. It means that in the header of the file with train dataset the **Target** and **Source** notions should be mentioned. Train dataset is stored in the file named `data.tsv`. `.tsv` stands for Tab Separated Values extension. Test dataset is stored in the `test.txt` file with sentences-to-translated written line by line.

## Dataset used

As a dataset the file from the following [site](https://opus.nlpl.eu/results/ru&en/corpus-result-table) was used. Its weight is about 14Mb. For the purpose of the example it will be split into two files: (i - `data.tsv`) all sentence but the last 500, (ii - `test.txt`) the last 500 sentences.

## Models used

Simple transformer model with [positional encoding](https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/).

## Deployment scheme

It is vitally important for the current programme to be cross-platform with no limitations about the OS and additional software. So, the solution in this case will be the Docker image. To extend the idea:

1. Launch the Docker image - obtain the *interactive* docker container.
2. Launch `train.py` module - given the *path to the train* dataset (`data.tsv`).
3. Launch `test.txt` module - give the *path to the test* dataset (`test.txt`).
4. Get the `output.txt` file with translated sentences.

The running OS for the current project will be Ubuntu.

## References

1. [Attention is all you need](https://arxiv.org/abs/1706.03762).
2. [Docker](https://www.docker.com).
3. [Dataset](https://opus.nlpl.eu/results/ru&en/corpus-result-table).