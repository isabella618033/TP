import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import interpolate
import pandas as pd
import numpy as np

"""
this was supposed to be ran after the cleanData.py

this piece of code is supposed to do interpolation for the missing data

for each of the attri in neededAttri, the interpolated data would be saved separately in a _filled.csv
"""

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
    print(attri)
    dfAttri = fullTimesInsatance.pivot_table(index = 'created', columns = 'brandID', values = attri,  aggfunc='mean')
    allIndex = dfAttri.index
    # print(len(allIndex))
    allIndex = pd.date_range(start= allIndex[0], end = allIndex[-1])
    # print(len(allIndex))
    allIndex.name = "created"
    dfAttriFilled = pd.DataFrame(index = allIndex)

    for brandID in dfAttri:
        TS = dfAttri[brandID].dropna()
        x = list(TS.index.values.astype('float'))
        y = list(TS)

        try:
            fittedFunction = InterpolatedUnivariateSpline(x, y, k=1)
        except Exception as e:
            continue

        fittedValue = fittedFunction(allIndex.values.astype('float'))
        TS_reindex = dfAttri[brandID]

        startDate = TS_reindex.first_valid_index()
        endDate = TS_reindex.last_valid_index()

        startIndex = allIndex.tolist().index(startDate)
        endIndex = allIndex.tolist().index(endDate)

        fittedValue[:startIndex] = np.nan
        fittedValue[endIndex+1:] = np.nan

        dfAttriFilled[brandID] = fittedValue


        # predPart = fittedValue[startIndex:lastIndex+1]
        # storedPart = dfAttri[brandID]
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

    dfAttriFilled.to_csv('./data/filledData/{}_filled.csv'.format(attri))
