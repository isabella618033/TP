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
import argparse
import pickle
from eval_util import suppress_stdout_stderr
import os.path
from statsmodels.tsa.vector_ar.var_model import VAR
import matplotlib.colors as mcolors

"""
this piece of code is not useful for the implementation
the evaluatin of this method was separated from evaluatePredictionMethods.py, because this method is targeting on multi-variate prediction method.
"""
class TSIndexEvaluation:
    def __init__(self, TSIndex, arg):
        self.TSIndex = TSIndex
        self.stage = arg.stage
        self.folder = arg.folder
        self.fittingDays = 60
        self.forecastingDays = 30
        self.methodName = "VAR"
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
        self.initDF()

    def initDF(self):
        """
        initialize the attriDF and attriDF_normalized to the class
        """
        self.allAttriDF = {}
        for attri in self.neededAttri:
            tempAttriDF = pd.read_csv('./data/filledData/{}_filled.csv'.format(attri)).set_index('created')
            tempAttriDF.index = pd.to_datetime(tempAttriDF.index)
            tempAttriDF.columns = [int(x) for x in tempAttriDF.columns]

            for TSIndex in tempAttriDF.columns:
                targetTS = tempAttriDF[TSIndex]
                mean = np.mean(targetTS)
                std = np.std(targetTS)
                tempAttriDF[TSIndex] = (targetTS-mean)/std
            self.allAttriDF[attri] = tempAttriDF

    def getMSE(self, brandID, TS):
        pointForecast = {}
        for attri in TS.columns:
            pointForecast[attri] = []
        for sampleIndex in range(len(TS)-(self.fittingDays + self.forecastingDays)):
            sys.stdout.write("\rsampleIndex: {}".format(sampleIndex))
            sys.stdout.flush()
            sampleTS = TS[sampleIndex : sampleIndex + self.fittingDays]
            sampleTS = sampleTS.reset_index()
            # try:

            sampleTS = sampleTS.drop("created", axis = 1)
            # print(sampleTS.head())
            model = VAR(endog=sampleTS.astype(float))
            model_fit = model.fit()
            sampleForecast = model_fit.forecast(model_fit.y, steps=self.forecastingDays)

            sampleForecast = pd.DataFrame(sampleForecast, columns = TS.columns)
            # print(sampleForecast.head())

            for attri in sampleForecast.columns:
                pointForecast[attri].append(sampleForecast[attri][self.forecastingDays-1])

            # if not isinstance(sampleForecast, pd.DataFrame):
            #     return
            # except Exception as e:
            #     print(e)
            #     print("sampleTS", sampleTS)
            #     sampleForecast = [np.nan]

        colors = list(mcolors.TABLEAU_COLORS) + ["brown", "chocolate", "darkkhaki", "olive", "darkolivegreen", "darkseagreen", "lightseagreen", "teal", "deepskyblue", "steelblue", "slategrey", "navy", "slateblue"]
        plt.figure(figsize=(20,10))
        for i, attri in enumerate(TS.columns):
            # print(attri, end = "\t\t")
            try:
                if len(pointForecast) > 0:
                    pointMSE =  mean_squared_error(TS[attri][-len(pointForecast[attri]):], pointForecast[attri])
                    # print("pointMSE", TS[attri][-len(pointForecast[attri]):], end = "\t")
                else:
                    pointMSE = np.nan

                if len(sampleForecast) > 0:
                    wholeMSE =  mean_squared_error(TS[attri][-len(sampleForecast[attri]):], sampleForecast[attri])

                    plt.plot(list(range(90)), TS[attri].reset_index().drop(["created"], axis = 1)[-90:], color = colors[i], label = attri)

                    wholeForecast_len = 30
                    wholeForecast_Plot = pd.concat([pd.Series([np.nan]*60),  sampleForecast[attri]])

                    plt.plot(list(range(90)), wholeForecast_Plot, '--' ,color = colors[i] ,label = attri)

                    print("wholeMSE", wholeMSE)


                else:
                    wholeMSE = np.nan
            except Exception as e:
                print(e)
                # print("TS", TS[-90])
                print(TS[attri][-len(pointForecast[attri]):])
                print(TS[attri][-len(sampleForecast[attri]):])

                print("pointForecast", pointForecast[attri])
                print("wholeForecast", sampleForecast[attri])
                pointMSE = np.nan
                wholeMSE = np.nan

            pointMSE_SD = np.std(pointForecast[attri])
            # self.exportResult_singleTS(TS, attri, sampleForecast[attri], pointForecast[attri], brandID, pointMSE_SD ,pointMSE, wholeMSE)

        plt.legend()
        plt.title("Model: VAR\nTSIndex: {}".format(self.TSIndex))
        folder = "evalVARResult"
        plt.savefig('./data/pic/{}/VAR_{}_.png'.format(folder, self.TSIndex))
        return

    def exportResult_singleTS(self, TS, attri, wholeForecast, pointForecast, brandID,pointMSE_SD, pointMSE, wholeMSE):
        wholeForecast = np.append([self.methodName, attri, self.fittingDays, self.forecastingDays], wholeForecast)
        pointForecast = np.append([self.methodName, attri, self.fittingDays, self.forecastingDays], pointForecast)

        df = pd.DataFrame([wholeForecast], index=[brandID])
        df.to_csv('./data/{}/wholeForecast_{}.csv'.format(self.folder,self.stage), mode='a', header=False, index = True)

        df = pd.DataFrame([ pointForecast], index=[brandID])
        df.to_csv('./data/{}/pointForecast_{}.csv'.format(self.folder,self.stage), mode='a', header=False, index = True)

        result = { "Model" : self.methodName,
                    "Attri": attri,
                  "FittingDays": self.fittingDays ,
                  "ForeCastingDays": self.forecastingDays,
                  "wholeMSE": wholeMSE,
                  "pointMSE": pointMSE,
                  "pointMSE_SD": pointMSE_SD,
                  "CIHit": np.nan
                  }
        df = pd.DataFrame(result, index=[brandID])
        df.to_csv('./data/{}/meta_{}.csv'.format(self.folder,self.stage), mode='a', header=False, index = True)

        return

    def runEvaluation(self):
        if self.stage == "rolling":
            TSList = [310,893,65,1223,2246, 2599, 2023, 2340, 601, 3050, 2285, 68]
            days = 150
        elif self.stage == "allTS":
            # TSList = [20]
            TSList = list(range(0,2000, 200))
            days = self.fittingDays + self.forecastingDays + 1
        else:
            raise Exception("wrong stage")

        for TSIndex in TSList:
            print("\nTSIndex", TSIndex)
            self.TSIndex = TSIndex
            # if  not  (TSIndex in self.checkedIndex.values):
            self.attriDF = pd.DataFrame()
            for attri in self.neededAttri:
                try:
                    targetTS_norm = self.allAttriDF[attri][TSIndex]
                    if targetTS_norm.dropna().shape[0] > 150:
                        targetTS_norm.name = attri
                        self.attriDF = self.attriDF.join(targetTS_norm.to_frame(), how="outer")
                except:
                    pass

            if self.TSIndex % 50 == 0:
                for i, attri in enumerate(self.attriDF):
                    plt.plot(self.attriDF[attri], label = attri)

                plt.legend()
                plt.savefig('./data/pic/TSAttris/TSAttris_{}.png'.format(self.TSIndex))
                plt.clf()
            try:
                TS = self.attriDF.copy()
                attriLen = {}
                for attri in TS.columns:
                    attriLen[attri] = len(TS[attri].dropna())
                attriLen = {k: v for k, v in sorted(attriLen.items(), key=lambda item: item[1])}
                attriLen = list(attriLen.keys())
                drop = 0
                TSTrial = TS.copy().dropna()

                while (len(TSTrial) < days) & (drop < len(attriLen)):
                    drop += 1
                    TSTrial = TS.copy()
                    TSTrial = TSTrial.drop(attriLen[:drop], axis = 1)
                    TSTrial = TSTrial.dropna()

                if len(TSTrial) >= days :
                    TSTrial = TSTrial[-days:]
                    self.getMSE(TSIndex,TSTrial)
            except Exception as e:
                print(e)
        return

    def setResults(self):
        col_names = [ "TSIndex","Model", "Attri", "FittingDays", "ForecastDays"] + list(range(150))
        wholeForecast = pd.read_csv('./data/{}/wholeForecast_{}.csv'.format(self.folder,self.stage), header=None, index_col = 0, names = col_names)
        pointForecast = pd.read_csv('./data/{}/pointForecast_{}.csv'.format(self.folder,self.stage), header=None, index_col = 0, names = col_names)
        self.wholeForecast = wholeForecast[(wholeForecast["Attri"] == attri) & (wholeForecast["Model"] == self.methodName)]
        self.pointForecast = pointForecast[(pointForecast["Attri"] == attri) & (pointForecast["Model"] == self.methodName)]

def initFile(folder, stage):
    try:
        os.remove('./data/{}/wholeForecast_{}.csv'.format(folder,stage))
        os.remove('./data/{}/pointForecast_{}.csv'.format(folder,stage))
        os.remove('./data/{}/meta_{}.csv'.format(folder,stage))
    except:
        pass

    df = pd.DataFrame(columns = ['TSIndex','Model',"Attri","FittingDays","ForecastingDays","wholeMSE",'pointMSE',"pointMSE_SD","CIHit"])
    df.to_csv('./data/{}/meta_{}.csv'.format(folder,stage), mode='a', header=True, index = False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cleanFile', action="store_true",help='clean the place where we store the result')
    parser.add_argument('--stageAllTS', action="store_true",help='if true, then all TS would be evaluated according to the last prediction, if false, the evaluation would be ran on sampled TS rolling based')

    args = parser.parse_args()

    if args.stageAllTS:
        args.stage = "allTS"
    else:
        args.stage = "rolling"

    args.folder = "evalVARResult"


    if not os.path.isdir("./data/pic/{}".format(args.folder)):
        os.mkdir("./data/pic/{}".format(args.folder))

    if not os.path.isdir("./data/pic/TSAttris"):
        os.mkdir("./data/pic/TSAttris")


    # if args.cleanFile:
    #     initFile(folder,args.stage)


    varEval = TSIndexEvaluation(20, args)
    varEval.runEvaluation()

    # metaDF = pd.read_csv('./data/{}/meta_{}.csv'.format(self.folder,self.stage))
    # self.checkedIndex = metaDF[(metaDF["Attri"] == attri) & (metaDF["Model"] == self.methodName) ]['TSIndex']
    # print(self.checkedIndex)
