import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
import random

def plot(attri, TSIndex, forecastDays, fittingDays, stage, folder, title="", pointPlot = False, findQuantile = np.nan):

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
    wholeForecast = pd.read_csv('./data/{}/wholeForecast_{}.csv'.format(folder, stage), index_col = 0, names = col_names)
    pointForecast = pd.read_csv('./data/{}/pointForecast_{}.csv'.format(folder, stage), index_col = 0, names = col_names)

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

    plt.xlabel('Time')
    plt.ylabel('Increase')
    plt.legend(title='FittingDays:')
    plt.grid(True)

    if findQuantile:
        plt.title('Whole estimation\nBrandID: {}\nAttribute: {}\nModel: {}\nQuantile: {}\nMSE: {}'.format(TSIndex, attri, findQuantile.model, round(findQuantile.q, 3), round(findQuantile.th,3)) )
        plt.savefig('./data/pic/{}/{}_{}_{}_{}_.png'.format(folder,attri,findQuantile.model, TSIndex, round(findQuantile.q, 3)))
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
        plt.xlabel('Time')
        plt.ylabel('Increase')
        plt.grid(True)
        plt.legend(title='FittingDays:')
        plt.show()

class findTSQuantile:
    def __init__(self,metaDF, model, attri, folder):
        self.model = model
        self.TargetDF = metaDF[(metaDF["Model"] == model) & (metaDF["Attri"] == attri)]
        self.folder = folder
        mse = self.TargetDF["pointMSE"]
        mse = mse[~np.isnan(mse)]


    def plot(self):
        for q in np.arange(0,1,0.2):
            self.q = q
            self.th = np.quantile(mse, q)
            filteredDF = self.TargetDF[self.TargetDF["pointMSE"] < self.th].sort_values(["pointMSE"], ascending=False)
            TSIndexs = filteredDF["TSIndex"]

            for i in TSIndexs[:5]:
                try:
                    plot(attri, i, forecastDays, fittingDays, stage, self.folder, "\n (Quantile = {})".format(q), False, self)
                    print()
                    print("Attri", attri)
                    print("Model", self.model)
                    print("quantile", self.q)
                    print("threshold", self.th)

                    print("TSIndex", i)
                    break
                except Exception as e:
                    print(e)
                    continue

def generalPlot(stage, metaDF, colors):
    plt.figure(figsize=(15,7))
    for i, model in enumerate(pd.unique(metaDF["Model"])):
        mse = metaDF[metaDF["Model"] == model][stage]
        mse = mse[~np.isnan(mse)]
        q = np.quantile(mse, 0.95)
        mse_filtered = mse[mse < q]
        try:
            mse_filtered.plot(kind='density', label = model, color = colors[i] )
            plt.axvline(x = np.quantile(mse, 0.8), color = colors[i] )
        except:
            pass

    if stage == "pointMSE":
        plt.title("MSE for the last result of each prediction \n (the verticle line is the 0.8 quantile)")
    else:
        plt.title("MSE for the whole prediction \n (the verticle line is the 0.8 quantile)")

    plt.xlabel("MSE")
    plt.xlim(0,np.quantile(mse, 0.9))
    plt.ylim(0)
    plt.legend()
    plt.savefig('./data/pic/predResultByModel.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--stageAllTS', action="store_true" ,help='set allTS mode or rolling mode')
    parser.add_argument('--evalProphetOnly', action="store_true", help='')
    parser.add_argument('--evalVAROnly', action="store_true", help='')


    args = parser.parse_args()

    if args.evalProphetOnly:
        model = "prophet"
        folder = "evalProphetResult"
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
                        # 'ig_hashtags_num',
                        'ig_posts_num',
                        'ig_avg_comments_num',
                        'ig_avg_likes_num',
                        ]

    elif args.evalVAROnly:
        model = "VAR"
        folder = "evalVARResult"
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
                        # 'ig_hashtags_num',
                        'ig_posts_num',
                        'ig_avg_comments_num',
                        'ig_avg_likes_num',
                        ]

    else:
        folder = "evalPredictionMethodResult"
        neededAttri = [
                        'fb_followers',
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

    metaDF = pd.read_csv('./data/{}/meta_{}.csv'.format(folder, stage))

    colors = list(mcolors.TABLEAU_COLORS) + ["brown", "chocolate", "darkkhaki", "olive", "darkolivegreen", "darkseagreen", "lightseagreen", "teal", "deepskyblue", "steelblue", "slategrey", "navy", "slateblue"]

    generalPlot("pointMSE", metaDF, colors)
    generalPlot("wholeMSE", metaDF, colors)

    print(metaDF.groupby(["Attri", "Model"]).mean())
    print(metaDF.groupby(["Attri", "Model"]).std())

    for  model in pd.unique(metaDF["Model"]):
        plt.figure(figsize=(15,7))
        attriQuantileDF = pd.DataFrame(columns = [ str(round(x,1)) for x in np.arange(0.2,1,0.1)])
        worstAttri = []
        for i, attri in enumerate(neededAttri):
            print(model, attri)
            # findTSQuantile(metaDF, model, attri, folder).plot()

            mse = metaDF[(metaDF["Model"] == model) & (metaDF["Attri"] == attri)]["pointMSE"]
            mse = mse[~np.isnan(mse)]
            mse = mse.apply(lambda x : x**0.5)
            try:
                q = np.quantile(mse, 0.95)
            except:
                print("ERROR in quantile")
                print(mse)
            mse_filtered = mse[mse < q]

            try:
                mse_filtered.plot(kind='density', label = attri, color = colors[i] )
                plt.axvline(x = np.quantile(mse, 0.8), color = colors[i] )
            except:
                pass    
            quanList = []
            for quan in np.arange(0.2,1,0.1):
                quanList.append(np.quantile(mse, quan))
            quanList = pd.Series(quanList, index = attriQuantileDF.columns, name = attri)
            attriQuantileDF = attriQuantileDF.append(quanList, ignore_index = True)

        plt.title("Error for the last result of each prediction \n (the verticle line is the 0.8 quantile)")
        plt.xlabel("Error \n(rooted MSE to make it comparable to the changes in the last 30 days)")
        plt.xlim(0,np.quantile(mse, 0.9))
        plt.ylim(0)
        plt.legend()
        plt.savefig('./data/pic/predResultByAttri_{}.png'.format(model))
        attriQuantileDF.index = neededAttri
        attriQuantileDF = attriQuantileDF.sort_values("0.7")
        print(attriQuantileDF)

        for attri in neededAttri:
          findTSQuantile(metaDF, model, attri, folder).plot()

    while(1):

        for i, attri in enumerate(neededAttri):
            print(i,attri)
        attri = neededAttri[int(input("Attri"))]

        for i, TSIndex in enumerate(TSList):
            print(i,TSIndex)

        TSIndex = TSList[int(input("TSIndex"))]
        plot(attri, TSIndex, forecastDays, fittingDays, stage, folder, pointPlot)
