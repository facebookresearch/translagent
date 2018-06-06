Emergent Translation in Multi-Agent Communication
==================================
PyTorch implementation of the models described in the paper [Emergent Translation in Multi-Agent Communication](https://arxiv.org/abs/1710.06922 "Emergent Translation in Multi-Agent Communication").

We present code for training and decoding both word- and sentence-level models and baselines, as well as preprocessed datasets.

Dependencies
------------------
### Python
* Python 2.7
* PyTorch 0.2
* Numpy

### GPU
* CUDA (we recommend using the latest version. The version 8.0 was used in all our experiments.)

### Related code
* For preprocessing, we used scripts from [Moses](https://github.com/moses-smt/mosesdecoder "Moses") and [Subword-NMT](https://github.com/rsennrich/subword-nmt "Subword-NMT").

Downloading Datasets
------------------
The original corpora can be downloaded from ([Bergsma500](https://www.clsp.jhu.edu/~sbergsma/LexImg/), [Multi30k](http://www.statmt.org/wmt16/multimodal-task.html), [MS COCO](http://cocodataset.org/#home)). For the preprocessed corpora see below.

| | Dataset |
| -------------      | -------------  |
| Bergsma500     | [Data](https://drive.google.com/open?id=1ZisXwMiev_0uscwUSqZ0QhhEmgPkAE0W) |
| Multi30k       | [Data](https://drive.google.com/open?id=14059L8cfNxxtR8jwRmOS45NmP0J7Rg9r) |
| MS COCO       | [Data](https://drive.google.com/open?id=14XUGgnXbt--rwfyM-raz9BKKJlnV1zXh) |

Before you run the code
------------------
1. Download the datasets and place them in `/data/word` (Bergsma500) and `/data/sentence` (Multi30k and MS COCO)
2. Set correct path in `scr_path()` from `/scr/word/util.py` and `scr_path()`, `multi30k_reorg_path()` and `coco_path()` from `/src/sentence/util.py`

Word-level Models
------------------

#### Running nearest neighbour baselines
```bash
$ python word/bergsma_bli.py 
```

#### Running our models
```bash
$ python word/train_word_joint.py --l1 <L1> --l2 <L2>
```

where `<L1>` and `<L2>` are any of {en, de, es, fr, it, nl}

Sentence-level Models
------------------

#### Baseline 1 : Nearest neighbour
```bash
$ python sentence/baseline_nn.py --dataset <DATASET> --task <TASK> --src <SRC> --trg <TRG>
```

#### Baseline 2 : NMT with neighbouring sentence pairs
```bash
$ python sentence/nmt.py --dataset <DATASET> --task <TASK> --src <SRC> --trg <TRG> --nn_baseline 
```

#### Baseline 3 : Nakayama and Nishida, 2017
```bash
$ python sentence/train_naka_encdec.py --dataset <DATASET> --task <TASK> --src <SRC> --trg <TRG> --train_enc_how <ENC_HOW> --train_dec_how <DEC_HOW>
```

where `<ENC_HOW>` is either `two` or `three`, and `<DEC_HOW>` is either `img`, `des`, or `both`.

#### Our models : 
```bash
$ python sentence/train_seq_joint.py --dataset <DATASET> --task <TASK>
```

#### Aligned NMT : 
```bash
$ python sentence/nmt.py --dataset <DATASET> --task <TASK> --src <SRC> --trg <TRG> 
```

where `<DATASET>` is `multi30k` or `coco`, and `<TASK>` is either 1 or 2 (only applicable for Multi30k).

Dataset & Related Code Attribution
------------------
* Moses is licensed under LGPL, and Subword-NMT is licensed under MIT License.
* MS COCO and Multi30k are licensed under Creative Commons.

Citation
------------------
If you find the resources in this repository useful, please consider citing:
```
@inproceedings{Lee:18,
  author    = {Jason Lee and Kyunghyun Cho and Jason Weston and Douwe Kiela},
  title     = {Emergent Translation in Multi-Agent Communication},
  year      = {2018},
  booktitle = {Proceedings of the International Conference on Learning Representations},
}
```
