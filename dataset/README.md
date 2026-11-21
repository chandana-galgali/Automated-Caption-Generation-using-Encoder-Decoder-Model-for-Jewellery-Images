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
2. Filter to earrings and necklaces, giving 5,407 captioned images.
3. Augment each image with geometric transforms (flips, rotations) and colour shifts, giving 32,190 images.
4. Train and validate on an 80/20 split (training resizes to 224×224 and applies ImageNet normalisation).

The raw image folders (~450 MB) are kept outside this repository because of GitHub size limits.
