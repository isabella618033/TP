import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import interpolate
import pandas as pd
import numpy as np
import sys
from sklearn.metrics import mean_squared_error
import os
import plotly
from fbprophet import Prophet
from tpaLSTM import TPALSTM
import torch
import argparse
import pmdarima as pm
import pickle
from eval_util import suppress_stdout_stderr
import os.path

'''
The organization of this .py is as follow

1. class attriEvaluation is for the evaluation of a single attribute with a given prediction function
2. class predictionMethodEvaluation is for a single prediction method.
    When the argument "evalProphetOnly" was set, it will hold multiple attriEvaluation object and loop through them
3. the prediction funcstions was set saparately
4.
'''

class attriEvaluation:
    def __init__(self, attri, parent):
        self.attri = attri
        self.stage = parent.stage
        self.folder = parent.folder
        self.fittingDays = 60
        self.forecastingDays = 30
        self.methodName = parent.methodName
        self.predictionFunction = parent.predictionFunction
        self.initDF()

    def initDF(self):
        """
        initialize the attriDF and attriDF_normalized to the class
        """
        self.attriDF = pd.read_csv('./data/filledData/{}_filled.csv'.format(self.attri)).set_index('created')
        self.attriDF.index = pd.to_datetime(self.attriDF.index)
        self.attriDF.columns = [int(x) for x in self.attriDF.columns]

        self.attriDF_normalized = self.attriDF.copy()
        for TSIndex in self.attriDF_normalized.columns:
            mean = np.mean(self.attriDF_normalized[TSIndex])
            std = np.std(self.attriDF_normalized[TSIndex])
            self.attriDF_normalized[TSIndex] = (self.attriDF_normalized[TSIndex]-mean)/std
        self.attriDF_normalized.columns = [int(x) for x in self.attriDF_normalized.columns]
        try:
            metaDF = pd.read_csv('./data/{}/meta_{}.csv'.format(self.folder,self.stage))
            self.checkedIndex = metaDF[(metaDF["Attri"] == self.attri) & (metaDF["Model"] == self.methodName) ]['TSIndex']
        except FileNotFoundError:
             self.checkedIndex = pd.Series([])
        print(self.checkedIndex)

    def getMSE_singleTS(self, brandID, TS):
        '''
        input
        brandID: int
        TS: pd.Series

        function
        given the TS, loop through the TS for excessive time points, and do prediction for each of them

        output
        wholeForecast: interpreted as the last forecast for the whole period of the ForeCastingDays
        pointForecast: the prediction made at each of the excessive time point for a single day output
        '''
        pointForecast = []
        sampleForecast = pd.DataFrame(columns = ['trend'])
        CICount = 0
        for sampleIndex in range(len(TS)-(self.fittingDays + self.forecastingDays)):
            sys.stdout.write("\rsampleIndex: {}".format(sampleIndex))
            sys.stdout.flush()
            sampleTS = TS[sampleIndex : sampleIndex + self.fittingDays]
            try:
                sampleForecast = self.predictionFunction(sampleTS, self.forecastingDays, self.attri, brandID)
                if not isinstance(sampleForecast, pd.DataFrame):
                    sampleForecast = pd.Series(sampleForecast)
            except Exception as e:
                print(e)
                print("sampleTS", sampleTS)
                sampleForecast = pd.Series([np.nan])
            if self.methodName == "prophet":
                pointForecast.append ( sampleForecast["yhat"][-1])
                if (sampleTS[-1] < sampleForecast["yhat_upper"][-1]) and (sampleTS[-1] > sampleForecast["yhat_lower"][-1]):
                    CICount += 1
                sampleForecast = sampleForecast["yhat"]
            else:
                pointForecast.append ( sampleForecast.iloc[-1] )


        try:
            if len(pointForecast) > 0:
                pointMSE =  mean_squared_error(TS[-len(pointForecast):], pointForecast)
            else:
                pointMSE = np.nan

            if len(sampleForecast) > 0:
                wholeMSE =  mean_squared_error(TS[-len(sampleForecast):], sampleForecast)
            else:
                wholeMSE = np.nan
        except Exception as e:
            print(e)
            print("TS", TS[-90])
            print("pointForecast", pointForecast)
            print("wholeForecast", sampleForecast)
            pointMSE = np.nan
            wholeMSE = np.nan



        pointMSE_SD = np.std(pointForecast)

        # plt.plot(TS.index.values.astype('float'), TS, 'ob', label = "org")
        # plt.plot(TS[-30:].index.values.astype('float'), sampleForecast, 'g', label = "extra")
        # plt.axvline(x = TS[-30:].index.values.astype('float')[0])
        # plt.legend()
        # plt.show()

        self.exportResult_singleTS(TS, sampleForecast, pointForecast, brandID, pointMSE_SD ,pointMSE, wholeMSE, CICount )
        return

    def exportResult_singleTS(self, TS, wholeForecast, pointForecast, brandID,pointMSE_SD, pointMSE, wholeMSE, CIRate):
        wholeForecast = np.append([self.methodName, self.attri, self.fittingDays, self.forecastingDays], wholeForecast)
        pointForecast = np.append([self.methodName, self.attri, self.fittingDays, self.forecastingDays], pointForecast)

        df = pd.DataFrame([wholeForecast], index=[brandID])
        df.to_csv('./data/{}/wholeForecast_{}.csv'.format(self.folder,self.stage), mode='a', header=False, index = True)

        df = pd.DataFrame([ pointForecast], index=[brandID])
        df.to_csv('./data/{}/pointForecast_{}.csv'.format(self.folder,self.stage), mode='a', header=False, index = True)

        result = { "Model" : self.methodName,
                    "Attri": self.attri,
                  "FittingDays": self.fittingDays ,
                  "ForeCastingDays": self.forecastingDays,
                  "wholeMSE": wholeMSE,
                  "pointMSE": pointMSE,
                  "pointMSE_SD": pointMSE_SD,
                  "CIHit": CIRate
                  }
        df = pd.DataFrame(result, index=[brandID])
        df.to_csv('./data/{}/meta_{}.csv'.format(self.folder,self.stage), mode='a', header=False, index = True)

        return

    def runEvaluation(self):
        if self.stage == "rolling":
            TSList = [310,893,65,1223,2246, 2599, 2023, 2340, 601, 3050, 2285, 68]
        elif self.stage == "allTS":
            TSList = self.attriDF.columns[150:300]
        else:
            raise Exception("wrong stage")

        for TSIndex in TSList:
            print("TSIndex", TSIndex)
            if  not  (TSIndex in self.checkedIndex.values):
                if self.stage == "rolling":
                    days = 150
                elif (self.stage == "allTS"):
                    days = self.fittingDays + self.forecastingDays + 1
                else:
                    raise Exception("wrong stage")
                try:
                    TS = self.attriDF_normalized.loc[:,TSIndex].dropna()[-days:]
                except Exception as e:
                    print(e)
                    continue
                self.getMSE_singleTS(TSIndex,TS)

        self.setResults()
        return

    def setResults(self):
        col_names = [ "TSIndex","Model", "Attri", "FittingDays", "ForecastDays"] + list(range(150))
        wholeForecast = pd.read_csv('./data/{}/wholeForecast_{}.csv'.format(self.folder,self.stage), header=None, index_col = 0, names = col_names)
        pointForecast = pd.read_csv('./data/{}/pointForecast_{}.csv'.format(self.folder,self.stage), header=None, index_col = 0, names = col_names)
        self.wholeForecast = wholeForecast[(wholeForecast["Attri"] == self.attri) & (wholeForecast["Model"] == self.methodName)]
        self.pointForecast = pointForecast[(pointForecast["Attri"] == self.attri) & (pointForecast["Model"] == self.methodName)]

class predictionMethodEvaluation:
    def __init__(self,methodName ,args, predictionFunction = np.nan):
        self.methodName = methodName
        self.predictionFunction = predictionFunction
        self.stage = args.stage

        if args.evalProphetOnly:
            self.folder = "evalProphetResult"
            self.neededAttri = [
                            'fb_likes',
                            'fb_followers',
                            'fb_posts_num',
                            'fb_avg_comments_num',
                            'fb_avg_likes_num',
                            'fb_avg_shares_num',
                            'ig_followings',
                            'ig_followers',
                            'ig_posts',
                            'ig_hashtags_num',
                            'ig_posts_num',
                            'ig_avg_comments_num',
                            'ig_avg_likes_num',
                            ]
        else:
            self.folder = "evalPredictionMethodResult"
            self.neededAttri = [
                            'fb_followers',
                            ]

        print(self.methodName)
        print(self.predictionFunction)
        print(self.stage)
        print(self.folder)

        for attri in self.neededAttri:
            tempClass = attriEvaluation(attri,self)
            if args.evalProphetOnly:
                tempClass.runEvaluation()
                del tempClass

    def runEvaluation(self):
        c = attriEvaluation("fb_followers", self)
        c.runEvaluation()

    def setResult(self):
        for attri in self.neededAttri:
            self.eval[attri].setResults()

def initFile(folder, stage):
    try:
        os.remove('./data/{}/wholeForecast_{}.csv'.format(folder,stage))
        os.remove('./data/{}/pointForecast_{}.csv'.format(folder,stage))
        os.remove('./data/{}/meta_{}.csv'.format(folder,stage))
    except:
        pass

    df = pd.DataFrame(columns = ['TSIndex','Model',"Attri","FittingDays","ForecastingDays","wholeMSE",'pointMSE',"pointMSE_SD","CIHit"])
    df.to_csv('./data/{}/meta_{}.csv'.format(folder,stage), mode='a', header=True, index = False)

    col_names = [ "TSIndex","Model", "Attri", "FittingDays", "ForecastDays"] + list(range(90))
    wholeForecast = pd.DataFrame( columns = col_names)
    wholeForecast.to_csv('./data/{}/wholeForecast_{}.csv'.format(folder,stage), index = False, header = True)
    wholeForecast.to_csv('./data/{}/pointForecast_{}.csv'.format(folder,stage), index = False, header = True)

def splineFitPredict(TS,forecastingDays,attri, brandID):
    x = list(TS.index.values.astype('float'))
    y = list(TS)
    fittedFunction = InterpolatedUnivariateSpline(x, y, k=1)
    predX = [(TS.index.values[-1]+np.timedelta64(i,'D')).astype('float') for i in range(1,1+forecastingDays)]
    pred = fittedFunction(predX)
    # predX = predX.index.values.astype('float')
    # pred = fittedFunction(predX)

    return pred

def prophetFitPredict(TS,forecastingDays,attri, brandID):
    print("\n",attri, brandID, end = "")
    pkl_path = "./ProphetModels/{}_{}.pkl".format(attri, brandID)

    TS = pd.DataFrame(TS).reset_index()
    TS.columns = ['ds','y']

    if os.path.isfile(pkl_path):
        pass
        # print("loading model")
        # with open(pkl_path, 'rb') as f:
        #     m = pickle.load(f)
        # f.close()
        return False
    else:
        print("building model")
        m = Prophet()
        with suppress_stdout_stderr() as s:
            m.fit(TS)
            # with open(pkl_path, "wb") as f:
            #     pickle.dump(m, f)
            #     f.close()

    future = m.make_future_dataframe(periods=forecastingDays)
    future.tail()
    forecast = m.predict(future).set_index('ds')
    del m
    return forecast

class MaxScaler:

    def fit_transform(self, y):
        self.max = np.max(y)
        return y / self.max

    def inverse_transform(self, y):
        return y * self.max

    def transform(self, y):
        return y / self.max

def tpaLSTMPredict(TS, forecastingDays, attri, brandID):

    modelPath = "./tpaLSTMSavedModel.pt".format(attri)

    with open(modelPath, 'rb') as f:
        model = torch.load(f)

    model.eval()
    scaler = MaxScaler()
    TS = np.array([TS])
    scale = TS[0][0]
    TS = torch.from_numpy(TS/scale).float()
    ypred = model(TS)
    ypred = ypred[0].detach().numpy()
    ypred = ypred*scale
    return ypred

def arimaPredict(TS, forecastingDays,attri, brandID):
    model = pm.auto_arima(TS, seasonal=True, m=12)
    forecasts = model.predict(forecastingDays)
    return forecasts

if __name__ == "__main__":
    """
    this code is not useful for the implementation
    this piece of code is supposed to test out all the prediction methods ans see which is the best,
    for the details of how to specify the arguments, please check the help function
    for the visualization of the reult, interpretPredictionResult.py has to be ran
    """


    parser = argparse.ArgumentParser()
    parser.add_argument('--cleanFile', action="store_true",help='clean the place where the prediction result was stored')
    parser.add_argument('--evalSpline', action="store_true", help='eval the spline prediction method')
    parser.add_argument('--evalProphet', action="store_true", help='eval prophe prediction method')
    parser.add_argument('--evaltpaLSTM',action="store_true",help='eval the tpaLSTM prediction method')
    parser.add_argument('--evalARIMA',action="store_true",help='eval the ARIMA prediction method')
    parser.add_argument('--stageAllTS',action="store_true",help='if true, then all the TS would be evaluated according to the last prediction, if false, the evaluation would be ran on sampled TS rolling based')
    parser.add_argument('--evalProphetOnly', action="store_true", help='this setting would override all the other settings. if true, all the attributes would be evaluated, if false, only the fb_follower would be evaluated')
    args = parser.parse_args()


    if args.stageAllTS:
        args.stage = "allTS"
    else:
        args.stage = "rolling"

    scale = "SCALE"

    if args.evalProphetOnly:
        folder = "evalProphetResult"
    else:
        folder = "evalPredictionMethodResult"

    if (not os.path.isdir("./data/{}".format(folder))) or (not os.path.isdir("./data/{}/meta_{}.csv".format(folder, args.stage))):
        os.mkdir("./data/{}".format(folder))
        initFile(folder,args.stage)

    if args.cleanFile:
        initFile(folder,args.stage)

    if args.evalProphetOnly:
        # https://github.com/facebook/prophet/issues/725
        prophetEval = predictionMethodEvaluation("prophet", args, prophetFitPredict)

    else:
        if args.evalProphet :
            prophetEval = predictionMethodEvaluation("prophet", args, prophetFitPredict)
            prophetEval.runEvaluation()

        if args.evalSpline :
            splineEval = predictionMethodEvaluation("spline", args, splineFitPredict)
            splineEval.runEvaluation()

        if args.evaltpaLSTM :
            tpaLSTMEval = predictionMethodEvaluation("tpaLSTM"+scale, args, tpaLSTMPredict )
            tpaLSTMEval.runEvaluation()

        if args.evalARIMA :
            ARIMAEval = predictionMethodEvaluation("arima", args, arimaPredict)
            ARIMAEval.runEvaluation()
