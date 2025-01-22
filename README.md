# Survival prediction in Titanic

Used [dataset](https://www.kaggle.com/c/titanic).

## Preprocessing
The datasets were missing some values. It was therefore important to handle these missing values properly: either by deletion (only those whose proportion was very small) or imputation (replacement).
After analysis of the data, it was decided to:
- delete row: Embarked (0.2% missing), Fare(0.2% missing)
- delete column: Cabin (77.1-78.2% missing), Ticket (not important)
- imputate with KNN: Age (19.9-20.6% missing)

Imputation was done using KNN algorithm from Impute module.

## Prediction methods

### Random Forest Model

### Logistic Regression

###  Gradient Boosted Trees

## Comparison of modelss
