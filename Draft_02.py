# logistic Regression

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#Apply a fix to the statsmodels library
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)


def getMasked(features):
    Masked_columns = {}
    print('now enter columns names to be masked one by one')
    while True:
        column = str(input('enter a column name, if no columns hit ''No'': ' )).strip()
        if column == 'No':
            break
        value_1 = str(input(f'And the first value in {column} is: '))
        value_2 = str(input(f'And the second value in {column} is: '))
        Masked_columns[column] = [value_1, value_2]
    return Masked_columns


def masking(raw_data, Masked_columns):
    for col, (v1, v2) in Masked_columns.items():
        raw_data[col] = raw_data[col].map({v1: 1, v2: 0})
    return raw_data


def LogisticRegression(fileName, target, features):
    raw_data = pd.read_csv(fileName)

    print(raw_data)
    Masked_columns = getMasked(features)

    data_masked = masking(raw_data, Masked_columns)
    print(f'data after masking\n {data_masked}')

    y = data_masked[target]
    x1= data_masked[features]

    x = sm.add_constant(x1)
    reg_log = sm.Logit(y,x)
    results_log = reg_log.fit()

    print(results_log.summary())

    print(results_log.pred_table())

    # Predict probabilities on training data
    train_pred_prob = results_log.predict(x)

    # Convert probabilities to binary class
    train_pred_class = (train_pred_prob >= 0.5).astype(int)

    # Calculate accuracy
    accuracy = (train_pred_class == y).mean()
    print(f"\n📊 Training Accuracy: {accuracy * 100:.2f}%")

    
    pred_file = str(input("\nEnter the file name for predictions: ")).strip()
    new_data = pd.read_csv(pred_file)

    # Apply same masking to prediction data
    new_data = masking(new_data, Masked_columns)

    # Prepare prediction features
    X_pred = sm.add_constant(new_data[features])

    # Predict probabilities and class
    new_data['Predicted_Prob'] = results_log.predict(X_pred)
    new_data['Predicted_Admitted'] = (new_data['Predicted_Prob'] >= 0.5).astype(int)

    print(new_data)

    # Save results
    output_name = input("\n💾 Enter file name to save predictions (without .csv): ")
    new_data.to_csv(output_name + ".csv", index=False)
    print(f"\n✅ Predictions saved to {output_name}.csv")


def get_info():
    x1 = str(input(f'Entetr the raw data file name: ')).strip()
    x2 = str(input('enter the target column name: ')).strip()
    x3 = [f.strip() for f in input(f'enter the features columns names, "-" seperated: ').split('-')]
    LogisticRegression(x1, x2, x3)

get_info()