| Experiment | Attacks | Detected | Detection rate | False alarms | FPR | Precision | F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| A: 2017->2018, all features | 362 | 0 | 0.0% | 98 | 0.068% | 0.0% | 0.000 | 0.921 |
| B: 2017->2018, shortcut features removed | 362 | 191 | 52.8% | 414 | 0.289% | 31.6% | 0.395 | 0.933 |
| C: 2018 train -> held-out 2018 | 109 | 108 | 99.1% | 12 | 0.028% | 90.0% | 0.943 | 0.998 |
| D: 2017+2018 train -> held-out 2018 | 109 | 109 | 100.0% | 10 | 0.023% | 91.6% | 0.956 | 1.000 |
| E: 2017->held-out 2018, shortcut removed | 109 | 60 | 55.0% | 129 | 0.300% | 31.7% | 0.403 | 0.933 |
