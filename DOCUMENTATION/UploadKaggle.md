# Kaggle Submission Upload Steps

This competition requires notebook submission. Follow these steps to submit.

## Files Ready

| File | Location | Size |
|------|----------|------|
| Models (zip) | `kaggle_models.zip` | 419 MB |
| Notebook | `kaggle_notebook_submission.ipynb` | - |

## Step 1: Create Kaggle Dataset for Models

1. Go to: https://www.kaggle.com/datasets?new=true
2. Name it: `forgery-detection-models`
3. Upload: `kaggle_models.zip` (it will auto-extract)
4. Visibility: **Private**
5. Click **Create**

The zip contains:
- `binary_classifier_best.pth` (32 MB)
- `highres_no_ela_v4_best.pth` (106 MB)
- `hard_negative_v4_best.pth` (106 MB)
- `high_recall_v4_best.pth` (106 MB)
- `enhanced_aug_v4_best.pth` (106 MB)

## Step 2: Create Kaggle Notebook

1. Go to competition: https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection
2. Click **Code** → **New Notebook**
3. Add data sources:
   - Click **Add data** → Search for your dataset `forgery-detection-models`
   - Competition data should already be attached
4. Copy code from `kaggle_notebook_submission.ipynb`

## Step 3: Update Model Path

In **Cell 3**, update the `MODEL_DIR` path:

```python
MODEL_DIR = Path('/kaggle/input/forgery-detection-models/kaggle_models')
```

## Step 4: Configure Notebook Settings

1. Click **Settings** (gear icon)
2. Set **Accelerator**: GPU T4 x2 or P100
3. Set **Internet**: Off (required for submission)
4. Set **Persistence**: Files only

## Step 5: Run and Submit

1. Click **Run All**
2. Wait for all cells to complete (~5-10 min)
3. Click **Submit** button (appears after successful run)

## Configuration Used

| Parameter | Value |
|-----------|-------|
| Classifier Threshold | 0.25 |
| Seg Threshold | 0.35 |
| Min Area | 300 |
| TTA | Enabled (4x flips) |
| Adaptive Threshold | Enabled |
| Ensemble | 4 models (mean aggregation) |

## Expected Results

Based on local validation:
- **Net Score**: ~2029 (TP - FP)
- **Recall**: 78.6%
- **FP Rate**: 5.6%
- **Forged Detected**: ~450 of 934 images
