import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
import random
"""
This piece of code is useless for implementation, it was just ment to visualize some distribution of the filled data
"""
def getDistribution(attr):

    attriDF = pd.read_csv('./data/filledData/{}_filled.csv'.format(attri)).set_index('created')
    attriDF.index = pd.to_datetime(attriDF.index)
    attriDF.columns = [int(x) for x in attriDF.columns]
    diff = []
    for TSIndex in attriDF.columns[0:2000]:
        TS = attriDF.loc[:,TSIndex]
        mean = np.mean(TS)
        std = np.std(TS)
        TS = (TS-mean)/std
        TS = TS.dropna()[-150:]
        try:
            diff.append(abs(np.mean(TS[-40]) - TS[-1]))
        except:
            pass
    return diff

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
colors = list(mcolors.TABLEAU_COLORS) + ["brown", "chocolate", "darkkhaki", "olive", "darkolivegreen", "darkseagreen", "lightseagreen", "teal", "deepskyblue", "steelblue", "slategrey", "navy", "slateblue"]

plt.figure(figsize=(15,7))

for i, attri in enumerate(neededAttri):
    print(attri)
    diff = getDistribution(attri)
    diff = pd.Series(diff)
    try:
        q = np.quantile(diff, 0.95)
    except:
        print("ERROR in quantile")
        print(diff)
    diff_filtered = diff[diff < q]
    diff_filtered.plot(kind='density', label = attri, color = colors[i] )
    plt.axvline(x = np.quantile(diff, 0.8), color = colors[i] )

plt.title("Distribution of changes in the last 30 days \n (the verticle line is the 0.8 quantile)")
plt.xlabel("Change")
plt.xlim(0,np.quantile(diff, 0.9))
plt.ylim(0)
plt.legend()
plt.savefig('./data/pic/last30DaysChanges.png')
