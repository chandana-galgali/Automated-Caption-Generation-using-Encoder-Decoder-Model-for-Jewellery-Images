# Dataset

| File | Contents |
|---|---|
| `labels/master_labels.csv` | **Source of truth**: one row per original photo (folder/file, description, type); 5,407 rows |
| `labels/combined_jewelry_dataset.xlsx` | Same data as the original spreadsheet |
| `labels/augmented_dataset.csv` | Index of the augmented images used by the original notebooks |
| `labels/dataset_labels.xlsx`, `naveen_dataset.xlsx`, `necklace_dataset.xlsx`, `cartier_catalog.xlsx` | Labels from the individual sources |
| `labels/DATASETS_INFO.xlsx` | Summary of the sources used |
| `test_set/` | 53 external web images + `labels.csv` / `test_descriptions.xlsx` (reference descriptions) |

**Sources:** AI Tool Pool Jewelry Vision, `imnaveenk/necklace`, `Bhargav1127/necklace-dataset-with-captions` (Hugging Face), plus manually collected earring photos (jhumkas, natural backgrounds, worn by a person).

**Preparation** (`training/prepare_dataset.py`):

1. Read `master_labels.csv` and the raw image folders.
2. Convert each image to RGB on a white background and resize it so the longer side is 512 px.
3. Group exact duplicate photos and identical descriptions together.
4. Split by group, stratified by type: 70% train, 15% validation and 15% test.

The result is 5,406 images: 3,548 earrings and 1,858 necklaces. 100 earring photos have no description and are used for classification only. Augmentation (random crop, flip, rotation and colour jitter) is applied on the fly to training images only, inside `training/run_all.py`.

**Download:** the prepared dataset (images + `labels.csv` with the fixed split + external test set, ~120 MB) is published on Kaggle: **[kaggle.com/datasets/chandanagalgali/jewellery-dataset](https://www.kaggle.com/datasets/chandanagalgali/jewellery-dataset)**.
It is not stored in this repository because of its size.