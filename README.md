# BERSON
Code for the paper "BERT-enhanced Relational Sentence Ordering Network".

We choose the [ROC dataset](https://github.com/sodawater/SentenceOrdering) as an example.  
Besides, we clean the original code and remove some useless modules which contribute little to the overall performance.


### Requirements
python3, pytorch 1.1.0


### Hyperparameter
The recommended hyperparameters for the current code of BERSON.  
```
ROC dataset:   batch=4, lr=2e-5, epoch=2,  coefficient=0.6  
AAN dataset:   batch=4, lr=5e-5, epoch=2,  coefficient=0.4  
NIPS dataset:  batch=8, lr=5e-5, epoch=20, coefficient=0.1  
SIND dataset:  batch=4, lr=2e-5, epoch=2,  coefficient=1.0  
Arxiv dataset: batch=4, lr=2e-5, epoch=2,  coefficient=0.8  
NSF dataset:   batch=4, lr=2e-5, epoch=2,  coefficient=0.4
```

### Train and evaluate
```
bash run.sh
```

### Citation
```
@inproceedings{cui2020bert,
  title={BERT-enhanced relational sentence ordering network},
  author={Cui, Baiyun and Li, Yingming and Zhang, Zhongfei},
  booktitle={Proceedings of EMNLP},
  pages={6310--6320},
  year={2020}
}
```

Some codes refer to the paper “Enhancing Pointer Network for Sentence Ordering with Pairwise Ordering Predictions” and "Fine-tune BERT for Extractive Summarization". Thanks!
