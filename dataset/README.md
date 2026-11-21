# Dataset

| File | Contents |
|---|---|
| `labels/combined_jewelry_dataset.xlsx` | Merged image list with descriptions and labels (earring / necklace) |
| `labels/dataset_labels.xlsx` | Labels from the primary source dataset |
| `labels/naveen_dataset.xlsx`, `labels/necklace_dataset.xlsx` | Supplementary necklace datasets (Hugging Face) |
| `labels/cartier_catalog.xlsx` | Additional catalogue descriptions |
| `labels/augmented_dataset.csv` | Final augmented dataset index (image, description, label) |
| `labels/DATASETS_INFO.xlsx` | Summary of the sources used |
| `test_set/` | 53 held-out web images + `test_descriptions.xlsx` with reference descriptions |

**Pipeline** (see `notebooks/01_*`):
1. Download and merge sources: AI Tool Pool Jewelry Vision, `imnaveenk/necklace`, `Bhargav1127/necklace-dataset-with-captions`, plus manually collected earring and necklace images.
2. Filter to earrings and necklaces: 5,307 captioned original photos (3,449 earrings, 1,858 necklaces).
3. Augment each photo ~6x with geometric transforms (flips, rotations) and colour shifts; `labels/augmented_dataset.csv` lists 31,842 images (the report quotes 32,190 for the final training set).
4. Train and validate on an 80/20 split (images resized to 224×224, ImageNet normalisation). The scripts in `training/` split by original photo to avoid augmented near-duplicates leaking into validation.

The raw image folders (~450 MB) are kept outside this repository because of GitHub size limits.
