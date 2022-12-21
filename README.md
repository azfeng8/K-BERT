# 6.8611 project
![](https://img.shields.io/badge/license-MIT-000000.svg)

This project is highly derived from the source code of the K-BERT paper: ["K-BERT: Enabling Language Representation with Knowledge Graph"](https://aaai.org/Papers/AAAI/2020GB/AAAI-LiuW.5594.pdf), which is implemented based on the [UER](https://github.com/dbiir/UER-py) framework.

The dataset and knowledge graphs we used are AI2 ARC and ConceptNet (version 5) respectively.

The files that we used to extract these data are under preprocess (they require a bit more work and documentation). The .tsv files are in a private GDrive currently.

We modified the knowledge injection mechanism and featurization of the words after switching to an English dataset + KG. These differences are mostly in knowgraph.py, and a bit of run_cls_bert.py.

We documented our work in Colab. 

Annie's link:

https://colab.research.google.com/drive/13uNnYRDnMIvd4ZLKtWTe4qnBF_8nkHEs?usp=sharing

Chao's link:

https://colab.research.google.com/drive/1FsVHgOTfia9nAHGFQQIyiDwgGAX5uAvJ?usp=share_link

## Requirements
You will need Python3.8 or higher.

Packages:
```
Pytorch >= 1.0
argparse == 1.1
re==2.2.1
numpy==1.23.3
```

## Prepare

The Colab file will mount access to the datasets and KGs.

The directory tree of K-BERT:
```
K-BERT
├── brain
│   ├── config.py
│   ├── __init__.py
│   ├── kgs
│   └── knowgraph.py
├── datasets (Colab)
│                                
│___preprocess
|   |
|   |__conceptnet5
|   |     |__english_only.py
|   |__easy.py
|
├── models (Colab)
├── outputs (Colab)
├── uer
├── README.md
├── requirements.txt
├── run_kbert_cls.py
└── run_kbert_ner.py
```
## K-BERT for text classification

### Classification example

Examples are provided in the Colab:

https://colab.research.google.com/drive/13uNnYRDnMIvd4ZLKtWTe4qnBF_8nkHEs?usp=sharing

Options of ``run_kbert_cls.py``:
```
useage: [--pretrained_model_path] - Path to the pre-trained model parameters.
        [--config_path] - Path to the model configuration file.
        [--vocab_path] - Path to the vocabulary file.
        --train_path - Path to the training dataset.
        --dev_path - Path to the validating dataset.
        --test_path - Path to the testing dataset.
        [--epochs_num] - The number of training epoches.
        [--batch_size] - Batch size of the training process.
        [--kg_name] - The name of knowledge graph, "HowNet", "CnDbpedia" or "Medical".
        [--output_model_path] - Path to the output model.
```

## Acknowledgement

If you use this code, please cite this paper:
```
@inproceedings{weijie2019kbert,
  title={{K-BERT}: Enabling Language Representation with Knowledge Graph},
  author={Weijie Liu, Peng Zhou, Zhe Zhao, Zhiruo Wang, Qi Ju, Haotang Deng, Ping Wang},
  booktitle={Proceedings of AAAI 2020},
  year={2020}
}
```