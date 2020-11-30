# Trend Prediction

To clean the raw data:
```
python cleanData.py
```

To fill the missing data:
```
python fillData.py
```

To build an evaluation of different models:
```
python evaluatePredictionMethods.py --scale

#use optional argument --evalSpline --evalProphet --evaltpaLSTM --evalARIMA to evaluate on the chosen method

#use optional argument --stageAllTS to choose the range of evaluation (only the endpoint of TS or the whole TS)
```

To view the result of evaluation:
```
python interpretResult.py --stageAllTS
```
