## LayoutGAN for bbox experiments (unofficial)

## Prerequisites

- Python 2.7
- Tensorflow 1.2.0
- [COCO API](https://github.com/cocodataset/cocoapi)

## Document layout generation
1. Use [PubLayNet dataset](https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/PubLayNet.html). Download `labels.tar.gz` and decompress it.
2. Run `python preprocess_doc.py` for preprocessing dataset.
3. Run `bash ./experiments/scripts/train_doc.sh` to train a model.

## Mobile app layout generation (WIP)
1. Use [Rico dataset](http://interactionmining.org/rico). Download `semantic_annotations.zip` and decompress it.
2. Run `python preprocess_mobile.py` for preprocessing dataset.
3. Run `bash ./experiments/scripts/train_mobile.sh` to train a model.

For more details, please see the original [README](ORIGINAL_README.md).
