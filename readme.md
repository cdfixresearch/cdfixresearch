# CDFixCode

This repository contains the code for **CDFixCode: Context-aware Dual Learning for Automated Program Repair** and the [Page](https://cdfixresearch.github.io/CDFixPage/) that has visualized data from table 1 to table 4.

# Experimental Results and Data: https://cdfixresearch.github.io/CDFixPage/
# Source Code: https://github.com/cdfixresearch/cdfixresearch#Instruction_to_Run_CDFix
# Demo: https://github.com/cdfixresearch/cdfixresearch#Demo

## Contents

1. [Website](#Website)
2. [Introduction](#Introduction)
3. [Dataset](#Dataset)
4. [Requirement](#Requirement)
5. [Instruction_to_Run_CDFix](#Instruction_to_Run_CDFix)
6. [Demo](#Demo)

## Website

We published the visualized experiment of our reasearch in https://cdfixresearch.github.io/CDFixPage/

## Introduction

Recent advances in deep learning (DL) have helped improve the
performance of the DL-based Automated Program Repair (APR)
approaches. The bug-fixing changes in APR often depend on the surrounding code context. Despite their successes, the state-of-the-art
DL-based APR approaches still have limitations in the integration
of code contexts in learning bug fixes. The limitations lead to the
ineffectiveness of those DL-based approaches in learning to auto-fix
context-dependent bugs. In this work, we conjecture that correct
learning of contexts can benefit the learning of code transformations and vice versa in APR. We introduce CDFix, a context-aware
dual learning APR model, which dedicates one model to learn the
bug-fixing code transformations (CTL) and another one to learn the
corresponding surrounding code contexts (CCL) for the transformations. Instead of cascading them, we train them simultaneously
with soft-sharing parameters via a cross-stitch unit to explicitly
model the impacts of contexts on fixing learning and vice versa.
We conducted several experiments to evaluate CDFix on three
different datasets: Defects4J [1] (395 bugs), Bugs.jar [36] (1,158
bugs), and BigFix [19] (+4.9M methods and 1.8M buggy ones). We
compared CDFix against several state-of-the-art DL-based APR
tools. Our results show that CDFix can fix 16.7%,12.1%, and 14.6%
more bugs than the best-performance DL-based baseline model
with only the top-1 patches in Defects4J, Bugs.jar, and BigFix, respectively. In Defects4J, it improves over the baseline models from
16.7%–194.7%. In Bugs.jar and BigFix, it fixes 26.4% and 27.7% of
the total fixed bugs that were missed by the best DL-based baseline.

## Dataset

### Preprocessed Dataset

We published our processed dataset at https://drive.google.com/file/d/1Vvw6r3kZQpeIniy71HqwlR6qia5whSxE/view?usp=sharing

Please create a folder named ```processed``` under the root folder of CDFix, download the dataset, unzip it and put all files in ```./processed``` folder.

### Use your own dataset

If you want to use your own dataset, please prepare the data as follow:

1. There are two data files including ```data_1.pt``` and ```dic.npy```

2. ```data_1.pt``` includes a list of fixing pairs and each pair stored in one list. Each pair contains four ```Tree``` object structure mentioned in ```main.py``` and four relevant value dictionary for each ```Tree``` object. The id in the dictionary matches the id in the ```Tree``` object. These four ```Tree``` object represent the following information:

	1> ```Tree_m```: AST for the buggy method
	
	2> ```Tree_mf```: AST for the fixed version of method
	
	3> ```Tree_s```: Subtree of AST for the buggy statement
	
	4> ```Tree_sf```: Subtree of AST for the fixed version statement
	
	```data_1.pt``` stores these trees with dictionaries as ```[Tree_m, Tree_mf, Tree_s, Tree_sf, dic_m, dic_mf, dic_s, dic_sf]```

3. ```dic.npy``` includes a numpy array and each line inside is a word embedding vector with lenght of 128 (changable). The index of the token is just the index of the numpy array. To transform to the readable results, please also prepare the dictionary reflecting the index with the real word tokens. In our experiments, we directly use index to check the accruacy without transform to real tokens.

## Requirement

Please check all required packages in the [requirement.txt](https://github.com/cdfixresearch/cdfixresearch/blob/main/requirement.txt) 

## Instruction_to_Run_CDFix

Download the CDFix source code from github and run ```main.py``` to see the result for our experiment. 

## Demo

For the testing purpose of running, please download our demo that contains the model for fixing one bug. Demo download: https://drive.google.com/file/d/1lE-Wqmbyd8eT2S0TOQt1TdUv3yb_KYY4/view?usp=sharing

Put ```encoder.pt```, ```decoder.pt``` and ```processed``` in the root folder of CDFix 

Change the path in ```demo_work.py``` if you are running on Linux system

then run ```demo_work.py``` to see the results.
