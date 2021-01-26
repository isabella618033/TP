# Trend Prediction

### initialize
To clean the data folder and set up spaces to save data and results 
```
./setUpDir.sh
```

### To clean the raw data:
```
python cleanData.py
```

### To fill the missing data:
```
python fillData.py
```

### To build an evaluation of different models for prediction (not for dev):
```
python evaluatePredictionMethods.py

#use optional argument --evalSpline --evalProphet --evalARIMA --evaltpaLSTM to evaluate on the chosen method
#use optional argument --stageAllTS to choose the range of evaluation (only the endpoint of TS or the whole TS)

python evaluateVAR.py

# this is for the VAR model only
```
The results will then be saved in the dir ./data/evalPredictionMethodResult

### To view the result of prediction evaluation (not for dev):
```
python interpretPredictionResult.py --stageAllTS
```
and the result for VAR model is in dir ./data/pic/evalVARResult

### Scoring and ranking (for dev)
scoring : resulted numbers for each brands are independent with any other brands
ranking : resulted numbers for each brands are relative to other brands  
```
python scoringAndRanking.py --clearFile --basicScoring
python scoringAndRanking.py --ranking
```
The results of the scoring and ranking is in ./data/brandsWithScoring.csv and ./data/brandsWithRanking.csv

## The Scoring
1. Prediction
  * The scoring of an attribute is the predicted value from Prophet of that attribute accoring to a 60 days history for a 30 days future

  * If daysBehind < 14 days, then it will predict for 30+daysBehind. Else, the value would be left as np.nan

2. Growth
  * The growth is from exponential smoothing

  * TSDiff_t = TS_t - TS_{t-1}
  * Growth_t = 0.04* TSDiff_t + (1-0.04)Growth_{t-1}
  * Where Growth_t is the cumulative growth at time t

## The Ranking
1. The de-correlation function between x and y
  * First, order [(x1,y1), (x2,y2)...] by x
  * Then, each y would be normalized by mean and SD of the 200 cloest values of y in [(x1,y1), (x2,y2)...]
  * Lastly, each x would be normalized by mean and SD of all x
  * return [(x1', y1'), (x2', y2')...]

2. The harmonicMean

  * H(values, weight, p) = (sum(weight*values^p)/sum(weight))^p
  * when p = -1, it is an AND gate
  * when p = 2, it is an OR gate

3. attriScoring
  * attriScoring is the scoring for 1 single attribute

  * attriScoring(attri) = H(de-cor(attri, attri_growth), [1,1], -1)
  * attri can be fb_followers, fb_avg_likes_num... etc.

4. Engagement = H([attriScoring(likes), attriScoring(comments), attriScoring(shares)], [0.6,0.2,0.2], -1)

5. Single social media scoring

  * FB/IG Score = H(attriScoring(attriScoring(followers), engagement), [1,1], -1)

6. Final score

  * Final Score = H([fb_score, ig_score], [1,1], 2)
