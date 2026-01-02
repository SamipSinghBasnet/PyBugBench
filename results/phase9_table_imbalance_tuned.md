| scope        | model        | variant        |   threshold |   accuracy |   precision |   recall |       f1 |   roc_auc |
|:-------------|:-------------|:---------------|------------:|-----------:|------------:|---------:|---------:|----------:|
| global       | RandomForest | baseline_tuned |        0.1  |   0.354993 |    0.352957 | 0.992078 | 0.520672 |  0.53945  |
| global       | RandomForest | balanced_tuned |        0.1  |   0.3548   |    0.352992 | 0.993071 | 0.520847 |  0.538648 |
| global       | LightGBM     | baseline_tuned |        0.25 |   0.357654 |    0.35414  | 0.994327 | 0.522269 |  0.546406 |
| global       | LightGBM     | balanced_tuned |        0.1  |   0.353118 |    0.353118 | 1        | 0.521933 |  0.546405 |
| django       | RandomForest | baseline_tuned |        0.1  |   0.572656 |    0.573332 | 0.996177 | 0.727794 |  0.559269 |
| django       | RandomForest | balanced_tuned |        0.1  |   0.572478 |    0.573273 | 0.99566  | 0.727609 |  0.561131 |
| django       | LightGBM     | baseline_tuned |        0.4  |   0.574078 |    0.574107 | 0.996693 | 0.728557 |  0.573322 |
| django       | LightGBM     | balanced_tuned |        0.3  |   0.574078 |    0.574107 | 0.996693 | 0.728557 |  0.57331  |
| scikit-learn | RandomForest | baseline_tuned |        0.1  |   0.270015 |    0.236484 | 0.950034 | 0.378701 |  0.569399 |
| scikit-learn | RandomForest | balanced_tuned |        0.3  |   0.300216 |    0.238577 | 0.90727  | 0.377806 |  0.562657 |
| scikit-learn | LightGBM     | baseline_tuned |        0.15 |   0.297633 |    0.243043 | 0.945532 | 0.38669  |  0.582937 |
| scikit-learn | LightGBM     | balanced_tuned |        0.4  |   0.337532 |    0.247483 | 0.896241 | 0.387863 |  0.582917 |