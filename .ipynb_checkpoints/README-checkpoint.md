# Module_14
Machine Learning Trading bot

![header](pics/header.png)
# Machine Learning Trading Bot
>Multiple supervised machine learning classifiers are used and tested; input features and parameters were adjusted in the analysis to enhance trading signals' accuracy and trading bot's ability to adapt to new data.

Analysis includes:

* [Establish a Baseline Performance](#establish-a-baseline-performance)

* [Tune the Baseline Trading Algorithm](#tune-the-baseline-trading-algorithm)

* [Optimize cumulative return](#optimize-cumulative-return-on-the-baseline-trading-algorithm)

* [Evaluate a New Machine Learning Classifier](#evaluate-a-new-machine-learning-classifier)

* [Create an Evaluation Report](#evaluation-report)
---


## Installation Guide

Before running the Jupyter notebook file, first, install the following dependencies in Terminal or Bash under the `dev` environment.

```python
  pip install pandas
  pip install matplotlib
  pip install -U scikit-learn
  pip install pathlib
  pip install numpy
```

---

## General Information
It is necessary to import all libraries and dependencies.
![first](pics/lib.png)
### Establish a Baseline Performance
-- After importing the original dataframe, calculate 'Actual Returns' based on closing price. 

```python
signals_df["Actual Returns"] = signals_df["close"].pct_change()
```
-- Generate trading signals using short- and long-window SMA values.

![second](pics/signal_gen.png)

```python
signals_df['Signal'] = 0.0
signals_df.loc[(signals_df['Actual Returns'] >= 0), 'Signal'] = 1
signals_df.loc[(signals_df['Actual Returns'] < 0), 'Signal'] = -1
```

-- Calculate the strategy returns and plot the original strategy returns

```python
signals_df['Strategy Returns'] = signals_df['Actual Returns'] * signals_df['Signal'].shift()
```
![third](pics/stra_returns.png)

-- Split data into training and testing datasets by dates

```python
training_begin = X.index.min()
training_end = X.index.min() + DateOffset(months=3)

X_train = X.loc[training_begin:training_end]
y_train = y.loc[training_begin:training_end]

X_test = X.loc[training_end+DateOffset(hours=1):]
y_test = y.loc[training_end+DateOffset(hours=1):]
```

-- Use the SVC classifier model from SKLearn's support vector machine (SVM) learning method to fit the training data and make predictions based on the testing data

```python
svm_model = svm.SVC()
svm_model = svm_model.fit(X_train_scaled, y_train)
svm_pred = svm_model.predict(X_test_scaled)
```
-- Generate classification report with the SVC model predictions
![fifth](pics/SVM_report_1.png)

-- Create a predictions dataframe
```python
predictions_df = pd.DataFrame(index=X_test.index)
predictions_df['Predicted'] = svm_pred
predictions_df['Actual Returns'] = signals_df['Actual Returns']
predictions_df['Strategy Returns'] = predictions_df['Actual Returns'] * predictions_df['Predicted']
```
-- Plot the actual returns versus the strategy returns of the SVM model
![sixth](pics/AvS.png)
