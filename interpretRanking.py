import pandas as pd
import scoringAndRanking as sr
import matplotlib.pyplot as plt

df = pd.read_csv("./data/brandsWithRanking.csv")
df = df.sort_values(['final_score'], ascending = False)

metaAttri = ['brandID', 'brand', 'final_score', 'fb_score', 'fb_followers_merged', 'fb_engagement', 'ig_score', 'ig_followers_merged', 'ig_engagement']
neededAttri = [
        'fb_followers',
        'fb_avg_comments_num',
        'fb_avg_likes_num',
        'fb_avg_shares_num',
        'ig_followers',
        'ig_avg_comments_num',
        'ig_avg_likes_num',
        ]
requiredAttri = metaAttri.copy()
for attri in neededAttri:
     requiredAttri = requiredAttri + [attri, attri + '_growth', attri+ '_merged']
df = df.loc[:100, requiredAttri]

df.to_csv('./data/sampledRanking.csv')


rankedBrands = pd.read_csv('./data/brandsWithRanking.csv')
rankedBrands = rankedBrands.sort_values(['final_score'], ascending = False)
count = 0
for index, row in rankedBrands.iterrows():
    if count > 10:
        break
    print("\n\n", row['brand'])
    brandID = row['brandID']
    brandName = row['brand']
    target = row[[isinstance(v, float) for v in row]].sort_values( ascending = False)
    for attri in target.keys():
        if attri in neededAttri:
            print(brandID, attri)
            self.plotTrend( int(brandID), attri, brandName )
    count += 1

ex = sr.scoringAndRanking()
ex.setUpBrands(False)
df = ex.allAttriDF.swaplevel(0,1,axis = 0)
for attri in neededAttri:
    plt.plot(df.loc[attri].count(), label = attri)
plt.ylabel("Num of observations")
plt.xlabel("Date")
plt.legend()
plt.title("Observation Count")
plt.show()

ex.checkDistributionOfRecent()
