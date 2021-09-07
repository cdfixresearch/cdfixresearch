# CDFixCode
This repository contains the code for **CDFixCode: Context-aware Dual Learning for Automated Program Repair**.

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

----------

## Contents
1. [Requirement](#Requirement)
2. [Instruction](#Instruction)


## Requirement

Please check all required packages in the [requirement.txt](https://github.com/cdfixresearch/cdfixresearch/blob/main/requirement.txt) 

## Instruction

Run ```main.py``` to see the result for our experiment. 


