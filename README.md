# Assignment 4: GAN-BERT Pytorch

NOTE: Our Assignemnt is split into 2 repositories. This repository is for performing Semi_Supervised Learning using GAN-BERT architecture. You can find the repository for Transfer Learning Experiments using Goemotions data at https://github.com/SamarthMM/GoEmotions-pytorch

# Abstract

Data has become ubiquitous in the modern day and age. However, the challenge lies in acquiring truthfully annotated high quality data. In this paper, we try to look into the challenge of limited labeled training data availability for NLP sentiment analysis tasks. We talk about the Sentiment Analysis task and it's broader usage in different fields and on varied datasets. We perform an extensive literature survey on the various model architectures used for emotion classification. We perform transfer learning by using a BERT base cased model on GoEmotions dataset for zero shot and one-shot fine tuning on twitter dataset. We then investigate the viability of using a GAN model as a semi supervised technique to leverage the presence of unlabeled data.

GAN-BERT is an extension of BERT which uses a Generative Adversarial setting to implement an effective semi-supervised learning schema. It allows training BERT with datasets composed of a limited amount of labeled examples and larger subsets of unlabeled material. 
GAN-BERT can be used in sequence classification tasks (also involving text pairs).

This code has been adapted from [crux82's implementation of GAN-BERT](https://github.com/crux82/ganbert-pytorch). The original README has been copied over to README_1.md


### Hyperparameters

| Parameter                    |      |
| -----------------------------| ---: |
| Learning rate (Generator)    | 5e-5 |
| Learning rate (Discriminator)| 5e-5 |
| Epochs            |   15 |
| Max Seq Length    |   50 |
| Batch size        |   64 |


### Requirements

- torch==1.4.0
- transformers==2.11.0
- attrdict==2.0.1

## How to Run

### GAN-BERT Model on Twitter:
```bash
#Change ratio of unlabeled to labeled examples "ratio" : "0.98 | 0.99"
#Change train_file to run on different train files "train_file" : "labeled_4000.csv | labeled_8000.csv"
python3 GANBERT_pytorch.py --unlabeled_ratio <ratio> --train_file <train_file.csv>


#For running on Twitter dataset with 8000 examples and 0.98 ratio and the output is saved into ganbert_8000_output.txt
python3 GANBERT_pytorch.py --unlabeled_ratio 0.98 --train_file labeled_8000.csv | tee ganbert_8000_output.txt
```

### GAN-BERT Model on GoEmotions:
```bash
#Change ratio of unlabeled to labeled examples "ratio" : "0.98 | 0.99"
#Change train_file to run on different train files "train_file" : "goemo_train_4000.csv | goemo_train_12000.csv"
python3 GANBERT_goemo.py --unlabeled_ratio <ratio> --train_file <train_file.csv>


#For running on Twitter dataset with 12000 examples and 0.98 ratio and the output is saved into ganbert_12000_output.txt
python3 GANBERT_goemo.py --unlabeled_ratio 0.98 --train_file goemo_train_12000.csv | tee ganbert_goemo_12000.txt
```

## Plotting Graphs:
```bash
python3 Results.py --file <output_file_name> --dataset <dataset.csv>

#For example the below command reads the output file that we generate above and the dataset from the csv to create plots for accuracy and number of epochs
python3 Results.py --file ganbert_8000_output.txt --dataset labeled_8000.csv
```

## Reference
https://aclanthology.org/2020.acl-main.191.pdf

## Acknowledgments

We would like to thank *Osman Mutlu* and *Ali Hürriyetoğlu* for their implementation of GAN-BERT in Pytorch that inspired our porting. 
You can find their initial repository at this [link](https://github.com/OsmanMutlu/Pytorch-GANBERT).
