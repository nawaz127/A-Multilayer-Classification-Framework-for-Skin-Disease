# Results & Discussion (Auto-draft)

## Top Models per Phase

| phase   | model         |   test_acc |   test_macroF1 |
|:--------|:--------------|-----------:|---------------:|
| ALL9    | swin_densenet |   0.968652 |       0.965263 |
| ALL9    | vit_resnet    |   0.959248 |       0.958373 |
| ALL9    | mobilenetv3   |   0.949843 |       0.945916 |
| L1      | resnet50      |   1        |       1        |
| L2      | densenet121   |   0.970803 |       0.969142 |
| L2      | resnet50      |   0.959854 |       0.958189 |

## Threshold Sensitivity (τ)

|   tau |      acc |   macroF1 |
|------:|---------:|----------:|
|  0.5  | 0.978056 |  0.976168 |
|  0.55 | 0.978056 |  0.976168 |
|  0.6  | 0.974922 |  0.973362 |
|  0.65 | 0.974922 |  0.973362 |
|  0.7  | 0.978056 |  0.976339 |
|  0.75 | 0.978056 |  0.976339 |
|  0.8  | 0.974922 |  0.973547 |

## Calibration (Temperature Scaling)

|   T |   acc_raw |   macroF1_raw |   ECE_raw |   acc_cal |   macroF1_cal |   ECE_cal |
|----:|----------:|--------------:|----------:|----------:|--------------:|----------:|
| 1.4 |  0.970803 |      0.969142 | 0.0187315 |  0.970803 |      0.969142 | 0.0267112 |

## ROC-AUC Tables

**auc_L1_resnet50.csv**

| phase   | model    |   AUC |
|:--------|:---------|------:|
| L1      | resnet50 |     1 |


**auc_ovr_ALL9_cnn.csv**

| class              |      AUC |
|:-------------------|---------:|
| Acne               | 0.889001 |
| Bacterial/Impetigo | 0.968827 |
| Eczema             | 0.86001  |
| Fungal             | 0.88968  |
| Normal             | 0.983455 |
| Pigmentation       | 0.999176 |
| Pox                | 0.932189 |
| Psoriasis          | 0.931076 |
| Scabies            | 0.824581 |
| macro              | 0.919777 |
| micro              | 0.919777 |


**auc_ovr_ALL9_mobilenetv3.csv**

| class              |      AUC |
|:-------------------|---------:|
| Acne               | 1        |
| Bacterial/Impetigo | 0.998773 |
| Eczema             | 0.998009 |
| Fungal             | 0.998595 |
| Normal             | 1        |
| Pigmentation       | 1        |
| Pox                | 0.998623 |
| Psoriasis          | 0.996842 |
| Scabies            | 0.995307 |
| macro              | 0.998461 |
| micro              | 0.998461 |


**auc_ovr_ALL9_swin_densenet.csv**

| class              |      AUC |
|:-------------------|---------:|
| Acne               | 1        |
| Bacterial/Impetigo | 1        |
| Eczema             | 0.99992  |
| Fungal             | 0.999344 |
| Normal             | 1        |
| Pigmentation       | 1        |
| Pox                | 0.99947  |
| Psoriasis          | 0.995536 |
| Scabies            | 0.998605 |
| macro              | 0.999208 |
| micro              | 0.999208 |


**auc_ovr_ALL9_vit_resnet.csv**

| class              |      AUC |
|:-------------------|---------:|
| Acne               | 1        |
| Bacterial/Impetigo | 1        |
| Eczema             | 0.99363  |
| Fungal             | 0.995973 |
| Normal             | 1        |
| Pigmentation       | 1        |
| Pox                | 1        |
| Psoriasis          | 0.992051 |
| Scabies            | 0.99518  |
| macro              | 0.997426 |
| micro              | 0.997426 |


**auc_ovr_L2_densenet121.csv**

| class              |      AUC |
|:-------------------|---------:|
| Acne               | 1        |
| Bacterial/Impetigo | 0.999855 |
| Eczema             | 0.999523 |
| Fungal             | 0.999554 |
| Pigmentation       | 1        |
| Pox                | 0.99912  |
| Psoriasis          | 0.998709 |
| Scabies            | 0.998051 |
| macro              | 0.999351 |
| micro              | 0.999351 |


**auc_ovr_L2_resnet50.csv**

| class              |      AUC |
|:-------------------|---------:|
| Acne               | 1        |
| Bacterial/Impetigo | 0.999855 |
| Eczema             | 0.997712 |
| Fungal             | 0.998439 |
| Pigmentation       | 1        |
| Pox                | 0.999246 |
| Psoriasis          | 0.999354 |
| Scabies            | 0.9988   |
| macro              | 0.999176 |
| micro              | 0.999176 |



## Statistical Significance (McNemar)
Insert chi-square and p-value.

## Explainability (Grad-CAM)
Insert 3–5 heatmaps and insights.

## Error Analysis
Summarize confusions and show misclassifications.

## Limitations & Future Work
Discuss domain shift and clinical context.