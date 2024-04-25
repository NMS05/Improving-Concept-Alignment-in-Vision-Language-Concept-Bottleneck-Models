
## Caltech-UCSD Birds (CUB)

---

### Data Preprocessing

- download the [CUB dataset](https://www.vision.caltech.edu/datasets/cub_200_2011/) and extract them to data/

- run data/preprocess_CUB.py to generate train/test splits for dataloader and to obtain ground-truth concept scores (required for estimating concept accuracy)

- the text concepts for the CUB dataset can be found in data/my_cub_concepts.txt

- run model/extract_concept_features.py to extract text concept features


### Experiments

- run clip_vl_cbm_train.py to train a vanilla vl-cbm model without concept supervision

- run css_vl_cbm_train.py to train the proposed CSS-VL-CBM model with concept semi-supervision


### Intervention

- After training, run confounding_classes/calculate_error_matrix.py to evaluate the trained CSS VL-CBM model

- Then use the notebook, confounding_classes/investigate_error_matrix.ipynb to identify the counfounding classes