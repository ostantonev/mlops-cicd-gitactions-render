# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model was developed during the training [*Machine Learning DevOps Engineer*](https://learn.udacity.com/nanodegrees/nd0821) and makes salary prediction using RandomForestClassifier from [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
The code and author information can be found on [Github](https://github.com/ostantonev/mlops-cicd-gitactions-render)
the model was trained with default parameters except of fixed random_state=42


## Intended Use
The model is not designed for productive use, but you can learn from developed solution.

## Training Data
The training dataset was downloaded from the [link](https://archive.ics.uci.edu/dataset/20/census+income) from training course.
We splitted the data 80% / 20% for training and validation. For some unclear reason we had issues in working with dashes in csv header for categorical fileds, therefore they were replaced by undescores. The CSV file contains is separated not by ',' but  by comma with spaces ', '
## Evaluation Data
The dataset is not well balanced, more infos in metrics section below

## Metrics
Following metrics were evaluated:
- **precision**=      0.75, 
- **recall**=      0.61, 
- **fbeta**=      0.67

Moreover, we evaluated the same metrics on different model slices (see the table below). From the table we can see that for some values we had less than 10 entries in the dataset, thus we cant expect any reasonable prediction for this case.
<pre>
category                      slice                              shape precision    recall     fbeta
----------------------------------------------------------------------------------------------------
workclass                     ?                                    359      0.76      0.64      0.69
workclass                     Federal-gov                          185      0.75      0.64      0.69
workclass                     Local-gov                            430      0.72      0.57      0.63
workclass                     Private                             4530      0.74      0.61      0.67
workclass                     Self-emp-inc                         239      0.74      0.74      0.74
workclass                     Self-emp-not-inc                     510      0.82      0.46      0.59
workclass                     State-gov                            255      0.76      0.71      0.73
workclass                     Without-pay                            5      1.00      1.00      1.00
education                     10th                                 171      1.00      0.30      0.46
education                     11th                                 233      1.00      0.18      0.31
education                     12th                                  78      0.60      0.38      0.46
education                     1st-4th                               32      1.00      1.00      1.00
education                     5th-6th                               81      1.00      0.50      0.67
education                     7th-8th                              120      1.00      0.25      0.40
education                     9th                                  123      1.00      0.00      0.00
education                     Assoc-acdm                           231      0.66      0.51      0.57
education                     Assoc-voc                            303      0.68      0.46      0.55
education                     Bachelors                           1040      0.75      0.71      0.73
education                     Doctorate                             77      0.80      0.86      0.83
education                     HS-grad                             2160      0.62      0.38      0.47
education                     Masters                              327      0.86      0.88      0.87
education                     Preschool                             11      1.00      1.00      1.00
education                     Prof-school                          117      0.90      0.93      0.91
education                     Some-college                        1409      0.71      0.52      0.60
marital_status                Divorced                             911      0.80      0.40      0.53
marital_status                Married-AF-spouse                      5      1.00      0.00      0.00
marital_status                Married-civ-spouse                  2939      0.74      0.64      0.69
marital_status                Married-spouse-absent                 95      0.67      0.29      0.40
marital_status                Never-married                       2144      0.81      0.42      0.55
marital_status                Separated                            212      1.00      0.25      0.40
marital_status                Widowed                              207      1.00      0.28      0.43
occupation                    ?                                    359      0.76      0.64      0.69
occupation                    Adm-clerical                         736      0.65      0.53      0.59
occupation                    Armed-Forces                           1      1.00      1.00      1.00
occupation                    Craft-repair                         841      0.57      0.39      0.46
occupation                    Exec-managerial                      812      0.80      0.74      0.77
occupation                    Farming-fishing                      190      1.00      0.41      0.58
occupation                    Handlers-cleaners                    261      0.75      0.30      0.43
occupation                    Machine-op-inspct                    397      0.67      0.40      0.50
occupation                    Other-service                        717      0.56      0.16      0.24
occupation                    Priv-house-serv                       33      1.00      1.00      1.00
occupation                    Prof-specialty                       802      0.80      0.74      0.77
occupation                    Protective-serv                      139      0.76      0.51      0.61
occupation                    Sales                                734      0.70      0.60      0.65
occupation                    Tech-support                         163      0.66      0.60      0.62
occupation                    Transport-moving                     328      0.75      0.37      0.49
relationship                  Husband                             2597      0.74      0.65      0.69
relationship                  Not-in-family                       1680      0.76      0.40      0.52
relationship                  Other-relative                       201      1.00      0.62      0.77
relationship                  Own-child                           1006      1.00      0.13      0.24
relationship                  Unmarried                            722      1.00      0.38      0.55
relationship                  Wife                                 307      0.75      0.59      0.66
race                          Amer-Indian-Eskimo                    74      0.78      0.50      0.61
race                          Asian-Pac-Islander                   206      0.69      0.67      0.68
race                          Black                                643      0.75      0.52      0.61
race                          Other                                 44      1.00      0.80      0.89
race                          White                               5546      0.75      0.61      0.67
sex                           Female                              2188      0.78      0.48      0.60
sex                           Male                                4325      0.74      0.63      0.68
native_country                ?                                    115      0.70      0.61      0.65
native_country                Cambodia                               7      0.00      0.00      0.00
native_country                Canada                                19      0.50      0.40      0.44
native_country                China                                 19      0.57      0.80      0.67
native_country                Columbia                              17      0.50      1.00      0.67
native_country                Cuba                                  24      0.80      0.67      0.73
native_country                Dominican-Republic                    14      0.00      1.00      0.00
native_country                Ecuador                                7      1.00      0.00      0.00
native_country                El-Salvador                           28      1.00      0.33      0.50
native_country                England                               17      0.60      0.75      0.67
native_country                France                                 6      0.50      0.50      0.50
native_country                Germany                               26      1.00      1.00      1.00
native_country                Greece                                 4      0.00      1.00      0.00
native_country                Guatemala                             15      1.00      0.00      0.00
native_country                Haiti                                  8      1.00      1.00      1.00
native_country                Holand-Netherlands                     1      1.00      1.00      1.00
native_country                Honduras                               3      1.00      1.00      1.00
native_country                Hong                                   6      0.67      0.67      0.67
native_country                Hungary                                5      1.00      0.67      0.80
native_country                India                                 17      0.60      0.60      0.60
native_country                Iran                                  10      1.00      1.00      1.00
native_country                Ireland                                8      1.00      0.50      0.67
native_country                Italy                                 14      1.00      0.50      0.67
native_country                Jamaica                               22      0.50      1.00      0.67
native_country                Japan                                 21      0.62      0.71      0.67
native_country                Laos                                   3      1.00      1.00      1.00
native_country                Mexico                               137      0.50      0.20      0.29
native_country                Nicaragua                             11      0.00      0.00      0.00
native_country                Outlying-US(Guam-USVI-etc)             4      1.00      1.00      1.00
native_country                Peru                                   7      1.00      0.00      0.00
native_country                Philippines                           37      0.71      0.91      0.80
native_country                Poland                                11      1.00      0.00      0.00
native_country                Portugal                               5      1.00      1.00      1.00
native_country                Puerto-Rico                           21      0.67      1.00      0.80
native_country                Scotland                               2      1.00      1.00      1.00
native_country                South                                 14      0.80      0.67      0.73
native_country                Taiwan                                12      0.75      1.00      0.86
native_country                Thailand                               3      1.00      0.50      0.67
native_country                Trinadad&Tobago                        2      1.00      1.00      1.00
native_country                United-States                       5799      0.75      0.60      0.67
native_country                Vietnam                                9      1.00      1.00      1.00
native_country                Yugoslavia                             3      1.00      1.00      1.00
</pre>
## Ethical Considerations
The model is trained on the training set available in public access. The rest API does not stores any data, but the request is logged, for the use with the real data the log information should be either removed or protected corresponding to GDPR rules. Moreover, the dataset is not well balanced (gender, race etc), the usage of race in the dataset is etically questionable.

## Caveats and Recommendations
The proposed solution can be used for training purposes after light code refactoring. 