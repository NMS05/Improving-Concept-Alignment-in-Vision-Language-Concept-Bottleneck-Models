
# Contrastive Semi-Supervised Vision-Language Concept Bottleneck Model (CSS VL-CBM)

This repo provides the PyTorch scripts for the paper - *Improving Concept Alignment in Vision-Language Concept Bottleneck Models*.

---

## Problem Statement

Recent approaches in Explainable AI automate the construction of Concept Bottleneck Models (CBM) with LLM and VLM. In this work, we investigate if the concept scores of VLM's like CLIP faithfully represent the visual truth for expert-defined concepts. Our investigations show that CLIP model has *"poor concept alignment"* with the following problems,

- **Low Concept Accuracy** - CLIP model has a low concept accuracy (concept scores donot faithfully represent the visual input) despite achieving a high classification performance.

- **Incorrect Concept Association** - For challenging classification problems like CUB, the CLIP model struggles to correctly attribute the fine-grain concepts to the visual input.


---

## Improving Concept Alignment with Contrastive Semi-Supervised (CSS) Learning

- To improve concept alignment, obtaining the supervisory concept labels for all training samples is cumbersome. 

- Hence, we propose Contrastive Semi-Supervised (CSS) learning approach that can improve the concept alignment in VL-CBM's with fewer labeled concept examples.

- CSS encourages consistent concept activations within the same class whislt discriminating (contrasting) them from the other classes + use a few labelled concept examples (semi-supervision) per class to align the concepts with the ground truth.

<img src="" width="700" height="350">

---

## Training

The scripts are provided individually for CUB, Rival-10 datasets and are organized as follows,

- data/ = contains Dataset() class

- model/ = contains the vanilla VL-CBM and CSS VL-CBM models

- clip_vl_cbm_train.py = trains vanilla VL-CBM (like [LF-CBM](https://github.com/Trustworthy-ML-Lab/Label-free-CBM), [LaBo](https://github.com/YueYANG1996/LaBo)).

- css_vl_cbm_train.py = trains the proposed model with CSS learning objective.


---

#### To Do

Scripts for AwA2 nd WBCAtt datasets to be updated!


