import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import interpolate
import pandas as pd
import numpy as np

fullTimesInsatance = pd.read_csv('./data/fullTimesInsatance_filtered.csv')
fullTimesInsatance.created = pd.to_datetime(fullTimesInsatance.created)
fullTimesInsatance = fullTimesInsatance.groupby(['brandID', 'created']).last().reset_index()

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

for attri in neededAttri:
    dfAttri = fullTimesInsatance.pivot_table(index = 'created', columns = 'brandID', values = attri,  aggfunc='mean')
    allIndex = dfAttri.index
    for brandID in dfAttri:
        TS = dfAttri[brandID].dropna()
        x = list(TS.index.values.astype('float'))
        y = list(TS)

        try:
            fittedFunction = InterpolatedUnivariateSpline(x, y, k=1)
        except Exception as e:
            continue

        fittedValue = fittedFunction(allIndex.values.astype('float'))
        TS_reindex = dfAttri[brandID].reset_index().iloc[:,1]
        startIndex = TS_reindex.first_valid_index()
        lastIndex = TS_reindex.last_valid_index()

        fittedValue[:startIndex] = np.nan
        fittedValue[lastIndex+1:] = np.nan

        dfAttri[brandID] = fittedValue


        predPart = fittedValue[startIndex:lastIndex+1]
        storedPart = dfAttri[brandID]
        # print(
        # len(x),lastIndex-startIndex ,
        # len(predPart[~np.isnan(predPart)] ),
        # len(storedPart[~np.isnan(storedPart)] ),
        # len(allIndex.values),
        # )

        # if len(predPart[~np.isnan(predPart)] ) > len(x)*1.5:
        # 
        #     plt.plot(list(dfAttri[brandID]), ".", label = "org")
        #     plt.plot(fittedValue, label = "fitted")
        #     plt.legend()
        #     plt.show()

    dfAttri.to_csv('./data/filledData/{}_filled.csv'.format(attri))
