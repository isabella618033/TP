import pandas as pd
import numpy as np
import glob
from evaluatePredictionMethods import predictionMethodEvaluation
import matplotlib.pyplot as plt
import argparse

def plot(attri, TSIndex, forecastDays, fittingDays, stage, title="", pointPlot = False, findQuantile = np.nan):

    attriDF = pd.read_csv('./data/filledData/{}_filled.csv'.format(attri)).set_index('created')
    attriDF.index = pd.to_datetime(attriDF.index)
    attriDF.columns = [int(x) for x in attriDF.columns]

    TS = attriDF.loc[:,TSIndex]

    mean = np.mean(TS)
    std = np.std(TS)
    TS = (TS-mean)/std
    TS = TS.dropna()[-150:]
    X_len = min(150, len(TS))

    col_names = [ "TSIndex","Model", "Attri", "FittingDays", "ForecastDays"] + list(range(150))
    wholeForecast = pd.read_csv('./data/evalPredictionMethodResult/wholeForecast_{}.csv'.format(stage), header=None, index_col = 0, names = col_names)
    pointForecast = pd.read_csv('./data/evalPredictionMethodResult/pointForecast_{}.csv'.format(stage), header=None, index_col = 0, names = col_names)

    wholeForecast = (wholeForecast[(wholeForecast["Attri"] == attri)
                            & (wholeForecast['ForecastDays'] == forecastDays)
                            & (wholeForecast['FittingDays'] == fittingDays)])

    wholeForecast_TSIndex = wholeForecast.loc[[TSIndex]]
    wholeForecast_TSIndex = wholeForecast_TSIndex.drop(['ForecastDays', "FittingDays", "Attri"], axis = 1).set_index('Model')

    plt.figure(figsize=(20,10))
    for Model in wholeForecast_TSIndex.index:
        wholeForecast_len = len (wholeForecast_TSIndex.loc[Model].dropna())
        wholeForecast_Plot = pd.concat([pd.Series([np.nan]*(X_len - wholeForecast_len)),  wholeForecast_TSIndex.loc[Model].dropna()])
        plt.plot(list(range(X_len)), wholeForecast_Plot , label = Model)

    plt.plot(list(range(X_len)), TS, linewidth=3)
    plt.axvline(x=[X_len-forecastDays])

    plt.xlabel('value')
    plt.ylabel('Time')
    plt.legend(title='FittingDays:')
    plt.grid(True)

    if findQuantile:
        plt.title('Whole estimation\nBrandID: {}\nModel: {}\nQuantile: {}\nMSE: {}'.format(TSIndex, findQuantile.model, round(findQuantile.q, 3), round(findQuantile.th,3)) )
        plt.savefig('./data/pic/{}_{}_{}_.png'.format(findQuantile.model, TSIndex, round(findQuantile.q, 3)))
    else:
        plt.title('Whole estimation, ForecastingDays: {}, BrandID: {}'.format(forecastDays, TSIndex) )
        plt.show()
    plt.clf()

    if pointPlot:

        pointForecast = (pointForecast[(pointForecast["Attri"] == attri)
                                & (pointForecast['ForecastDays'] == forecastDays)
                                & (pointForecast['FittingDays'] == fittingDays)])
        pointForecast_TSIndex = pointForecast.loc[[TSIndex]]
        pointForecast_TSIndex = pointForecast_TSIndex.drop(['ForecastDays', "FittingDays", "Attri"], axis = 1).set_index('Model')

        plt.figure(figsize=(20,10))
        for Model in wholeForecast_TSIndex.index:
                pointForecast_len = len(pointForecast_TSIndex.loc[Model].dropna())
                pointForecast_Plot = pd.concat([pd.Series([np.nan]*(X_len - pointForecast_len)),  pointForecast_TSIndex.loc[Model].dropna()[-100:]])
                plt.plot(list(range(X_len)), pointForecast_Plot , label = Model)

        plt.plot(list(range(X_len)), TS, label = "Origional")
        plt.axvline(x=[150-forecastDays])
        plt.title('Rolling estimation, ForecastingDays: {}, BrandID: {}'.format(forecastDays, TSIndex) + title)
        plt.xlabel('Increase')
        plt.ylabel('Time')
        plt.grid(True)
        plt.legend(title='FittingDays:')
        plt.show()

class findTSQuantile:
    def __init__(self,metaDF, model):
        self.model = model
        self.TargetDF = metaDF[metaDF["Model"] == model]
        mse = self.TargetDF["pointMSE"]
        mse = mse[~np.isnan(mse)]


    def plot(self):
        for q in np.arange(0,1,0.2):
            self.q = q
            self.th = np.quantile(mse, q)
            filteredDF = self.TargetDF[self.TargetDF["pointMSE"] < self.th].sort_values(["pointMSE"], ascending=False)
            # print("threshold", self.th)
            # print(filteredDF)

            TSIndexs = filteredDF["TSIndex"]
            # print("selected TSIndex", TSIndexs)



            for i in TSIndexs[:10]:
                try:
                    plot("fb_followers", i, forecastDays, fittingDays, stage, "\n (Quantile = {})".format(q), False, self)
                    print()
                    print("Model", self.model)
                    print("quantile", self.q)
                    print("threshold", self.th)

                    print("TSIndex", i)
                    break
                except Exception as e:
                    print(e)
                    continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--attri', type=str, required=True,help='location of the data file')
    # parser.add_argument('--TSIndex', type=int, required=True,help='location of the data file')
    parser.add_argument('--stageAllTS', action="store_true" ,help='location of the data file')

    args = parser.parse_args()

    neededAttri = [
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

    TSList = [80,123,220,13,310,893,65,1223,2246, 2599, 2023, 2340, 601, 3050, 2285, 68]

    forecastDays = 30
    fittingDays = 60

    if args.stageAllTS:
        stage = "allTS"
        pointPlot = False
    else:
        stage = "rolling"
        pointPlot = True

    print("stage", stage)

    metaDF = pd.read_csv('./data/evalPredictionMethodResult/meta_{}.csv'.format(stage))
    colors = ["C0", "C1", "C2", "C3"]
    for i, model in enumerate(pd.unique(metaDF["Model"])):
        mse = metaDF[metaDF["Model"] == model]["pointMSE"]
        mse = mse[~np.isnan(mse)]
        q = np.quantile(mse, 0.95)
        mse_filtered = mse[mse < q]
        mse_filtered.plot(kind='density', label = model, color = colors[i] )
        plt.axvline(x = np.quantile(mse, 0.8), color = colors[i] )

    plt.title("MSE for the last result of each prediction \n (the verticle line is the 0.8 quantile)")
    plt.xlabel("MSE")
    plt.xlim(0,np.quantile(mse, 0.9))
    plt.ylim(0)
    plt.legend()
    plt.show()

    for i, model in enumerate(pd.unique(metaDF["Model"])):
        mse = metaDF[metaDF["Model"] == model]["wholeMSE"]
        mse = mse[~np.isnan(mse)]
        q = np.quantile(mse, 0.95)
        mse_filtered = mse[mse < q]
        mse_filtered.plot(kind='density', label = model, color = colors[i] )
        plt.axvline(x = np.quantile(mse, 0.8) ,color = colors[i] )

    plt.title("MSE for the prediction of the whole TS \n (the verticle line is the 0.8 quantile)")
    plt.xlabel("MSE")
    plt.xlim(0,np.quantile(mse, 0.9))
    plt.ylim(0)
    plt.legend()
    plt.show()

    print(metaDF.groupby(["Attri", "Model"]).mean())
    print(metaDF.groupby(["Attri", "Model"]).std())

    for i, model in enumerate(pd.unique(metaDF["Model"])):
        findTSQuantile(metaDF, model).plot()


    while(1):

        for i, attri in enumerate(neededAttri):
            print(i,attri)
        attri = neededAttri[int(input("Attri"))]

        for i, TSIndex in enumerate(TSList):
            print(i,TSIndex)

        TSIndex = TSList[int(input("TSIndex"))]

        plot(attri, TSIndex, forecastDays, fittingDays, stage, pointPlot)
