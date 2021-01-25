import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import statistics
import argparse
from scipy.optimize import curve_fit
from fbprophet import Prophet
from eval_util import suppress_stdout_stderr

def prophetFitPredict(seq, forecastingDays):
    TS = pd.DataFrame(seq).reset_index()
    TS.columns = ['ds', 'y']
    m = Prophet()
    m.fit(TS)
    future = m.make_future_dataframe(periods = forecastingDays)
    future.tail()
    forecast = m.predict(future).set_index('ds')
    del m
    return max(0, forecast.iloc[-1]['yhat'])

class scoringAndRanking:
    def __init__(self):
        self.expFactor = 0.04
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
                'ig_posts_num',
                'ig_avg_comments_num',
                'ig_avg_likes_num',
                ]

        self.neededAttriGrowth = [
                'fb_likes',
                'fb_followers',
                'fb_avg_comments_num',
                'fb_avg_likes_num',
                'fb_avg_shares_num',
                'ig_followers',
                'ig_avg_comments_num',
                'ig_avg_likes_num',
                ]

        self.neededAttri_FB = [
                    'fb_likes',
                    'fb_avg_comments_num',
                    'fb_avg_likes_num',
                    'fb_avg_shares_num',

                    'fb_followers_growth',
                    'fb_likes_growth',
                    'fb_avg_comments_num_growth',
                    'fb_avg_likes_num_growth',
                    'fb_avg_shares_num_growth',
                    ]

        self.neededAttri_IG = [
                    'ig_followings',
                    'ig_avg_comments_num',
                    'ig_avg_likes_num',

                    'ig_followers_growth',
                    'ig_avg_comments_num_growth',
                    'ig_avg_likes_num_growth',
                    ]

    def readCSV(self, attri):
        """
        helper function for getting self.allAttriDF
        """
        df = pd.read_csv("./data/filledData/{}_filled.csv".format(attri)).set_index('created')
        df.index = pd.to_datetime(df.index)
        df.columns = [int(x) for x in df.columns]
        return df.T

    def expGrowth(self,seq):
        """
        get the growth rate of the given sequence
        """
        diff = np.diff(seq)
        length = len(diff)
        weightedGrowth = [val*self.expFactor*(1-self.expFactor)**(length-index-1) for index, val in enumerate(diff) ]
        weightedGrowth = [x for x in weightedGrowth if not math.isnan(x)]

        result = sum(weightedGrowth)
        if result == np.nan:
            print(result)
        else:
            return result

    def setUpBrands(self, renew):
        """
        if renew = true, it will clean up the brandsWithScoring.CSV
        if false, the scoredBrands and rankedBrands would set up accoring to the saved CSV

        result
        scoredBrands: the scored brands (numbers independent with any other brands)
        rankedBrands: the ranked brands (numbers relative to other brands)
        allAttriDF: the very origional data for each attri

        """
        self.brands = pd.read_csv("./data/uniqueBrands.csv").set_index('brandID', drop = False)
        df = pd.DataFrame()
        newColumns = self.brands.columns.tolist() + self.neededAttri + [x + '_growth' for x in self.neededAttriGrowth]
        for col in newColumns:
            df[col] = ''

        df.to_csv('./data/brandsWithScoring_temp.csv', header=True, index = False)

        if renew:
            self.scoredBrands = df.copy()
            df.to_csv('./data/brandsWithScoring.csv', header=True, index = False)
        else:
            scoredBrands = pd.read_csv("./data/brandsWithScoring.csv").sort_values(['brandID'])
            scoredBrands["brandID"] = scoredBrands["brandID"].astype('int')
            scoredBrands = scoredBrands.set_index('brandID', drop = False)
            scoredBrands = scoredBrands.sort_index()
            self.scoredBrands = scoredBrands

            rankedBrands = pd.read_csv("./data/brandsWithRanking.csv").sort_values(['brandID'])
            rankedBrands["brandID"] = rankedBrands["brandID"].astype('int')
            rankedBrands = rankedBrands.set_index('brandID', drop = False)
            rankedBrands = rankedBrands.sort_index()
            self.rankedBrands = rankedBrands

        frames = [self.readCSV(attri) for attri in self.neededAttri]
        allAttriDF = pd.concat(frames, keys = self.neededAttri)
        allAttriDF = allAttriDF.swaplevel(0,1,axis=0)

        self.allAttriDF = allAttriDF

    def plotTrend(self, brandID,attri, brandName = ""):
        """
        used when wanting to visualize/trace back some trend
        """

        target = self.allAttriDF.loc[brandID].reindex().T
        plt.plot(target.loc[ pd.Timestamp('2019-12-01T08') : pd.Timestamp('2020-03-01T08'),attri])
        plt.title("{}  {}  {}".format( brandName, attri, brandID))
        plt.savefig("./data/pic/checkRanking/{}_{}.png".format(brandID, attri))
        plt.clf()

    def checkDistributionOfRecent(self):
        """
        for visualization purpose, not for dev
        """
        brands = self.brands
        scoredBrands = self.scoredBrands
        frames = [self.readCSV(attri) for attri in self.neededAttri]
        allAttriDF = pd.concat(frames, keys = self.neededAttri)
        allAttriDF = allAttriDF.swaplevel(0,1,axis=0)

        availAttriIndex = [x[0] for x in allAttriDF.index]
        metaCol = brands.columns
        mostRecentRecord = allAttriDF.loc[1].reindex().T.iloc[-1,:].name
        daysBehindDis = {}
        seqLengthDis = {}
        fitCount = {}
        for attri in self.neededAttri:
            daysBehindDis[attri] = []
            seqLengthDis[attri] = []
            fitCount[attri] = 0

        for brandID in brands.index:
            if brandID in availAttriIndex:
                brandDF = allAttriDF.loc[brandID].reindex().T
                for attri in brandDF.columns:
                    attriSeq = brandDF.loc[:,attri].dropna()
                    daysBehind = (mostRecentRecord - attriSeq.index[-1]).days
                    daysBehindDis[attri].append(daysBehind)
                    seqLengthDis[attri].append(len(attriSeq))
                    if (daysBehind <= 14) & (len(attriSeq)>=60):
                        fitCount[attri] += 1
        for attri in self.neededAttri:
            plt.hist(seqLengthDis[attri],histtype = 'step', cumulative = -1, label = attri, )
        plt.title("length of sequence")
        plt.xlabel("length of sequence")
        plt.ylabel("number of brands")
        plt.legend()
        plt.show()
        plt.clf()

        for attri in self.neededAttri:
            plt.hist(daysBehindDis[attri],histtype = 'step', cumulative = -1, label = attri)
        plt.title("days behind")
        plt.xlabel("days behind")
        plt.ylabel("number of brands")
        plt.legend()
        plt.show()
        plt.clf()

        for attri in self.neededAttri:
            print(attri, fitCount[attri])

    def basicScoring(self):
        """
        for each of the brands and their attributes, find the predicted scoring and their growth
        """
        brands = self.brands
        scoredBrands = self.scoredBrands

        allAttriDF = self.allAttriDF
        minnedIndex = scoredBrands['brandID']
        availAttriIndex = [x[0] for x in allAttriDF.index]

        metaCol = brands.columns
        #mostRecentRecord = allAttriDF.loc[1].reindex().T.iloc[-1,:].name
        mostRecentRecord = pd.Timestamp('2020-03-01T08')
        for brandID in brands.index:
            if (brandID in availAttriIndex) & (brandID not in minnedIndex):
                brandDF = allAttriDF.loc[brandID].reindex().T
                scoredBrands.loc[brandID, metaCol] = brands.loc[brandID]
                print(brandID, end = " ")
                for attri in brandDF:
                    attriSeq = brandDF.loc[:mostRecentRecord,attri].dropna()
                    attriSeq = brandDF.loc[:,attri].dropna()
                    if (len(attriSeq) > 60):
                        daysBehind = (mostRecentRecord - attriSeq.index[-1]).days
                        if (daysBehind <= 14):
                            scoredBrands.loc[brandID, attri] = prophetFitPredict(attriSeq.iloc[-60:], daysBehind + 30)
                            if attri in self.neededAttriGrowth:
                                scoredBrands.loc[brandID, "{}_growth".format(attri)] = self.expGrowth(attriSeq)
                scoredBrands.loc[brandID].to_frame().T.to_csv('./data/brandsWithScoring_temp.csv', mode = 'a', header = False, index = False)

        dfA = pd.read_csv("./data/brandsWithScoring_temp.csv", index_col = False).set_index('brandID', drop = False)
        dfB = pd.read_csv("./data/brandsWithScoring.csv", index_col = False).set_index('brandID', drop = False)
        self.scoredBrands = dfA.combine_first(dfB)
        print(dfA.head())
        print(dfB.head())
        print(self.scoredBrands)
        self.scoredBrands.to_csv('./data/brandsWithScoring.csv', header = True, index = False)

    def sigmoid(self,x):
        """
        simple sigmoid function
        """
        try:
            return 1/(1+math.exp(-x))
        except:
            return np.nan

    def shiftedSigmoid(self, x):
        """
        shifting the x of sigmoid function by 1,
        cause we want to focus on some high-performing brands, doing so can make differentiations for these brands
        """
        return self.sigmoid(x-1)

    def harmonicMean(self, row, p =-1, weight = []):
        """
        calculate the harmonic mean of the given row
        when p = -1, it is AND gate
        when p = 2, it is OR gate
        the weight can be adjusted when calling this function
        """
        ele = row.copy().tolist()

        if (len(ele) == 0) or all ([np.isnan(x) for x in ele]):
            return np.nan
        ele = [self.shiftedSigmoid(0) if np.isnan(x) else x for x in ele]
        if len(weight) == 0:
            weight = [1 for e in ele]
        aggri = 0
        n = sum(weight)
        for i, e in enumerate(ele):
            if not math.isnan(e):
                aggri += weight[i]*e**p

        return (aggri/n)**(1/p)

    def rankAndDecorrelationAttri(self,x,y):
        """
        to decorelate:
        1. order the target dataFrame by x
        2. for each y, it would be normalized according to the 200 cloest values of y according to the ordered dataFrame
        3. x would also be normalized by itself

        to rank:
        harmonic mean (AND) for the resulted x and y from the above
        """
        print(len(x.dropna()), len(y.dropna()))
        target = pd.DataFrame( {'x':x, 'y':y})
        orgTarget = pd.DataFrame(index = target.index)
        target = target.dropna(how = 'all')
        if len(target) == 0:
            return pd.DataFrame([np.nan]*len(x), index = range(len(x)))[0]
        window = min(200, len(target))
        target = target.sort_values(by = ['x'])
        stat = target.iloc[:window].loc[:,'y']
        stat = np.array(stat)
        score = []
        means = []
        sds = []
        for i, yValue in enumerate(target['y']):
            if not math.isnan(yValue):
                if (i > window) & (i < len(target)-window):
                    stat = stat[1:]
                    stat = np.append(stat, yValue)
                upperBound = np.quantile(stat, 0.97, interpolation = "higher")
                lowerBound = np.quantile(stat, 0.03, interpolation = "lower")
                targetStat = stat[(stat <= upperBound) & (stat >= lowerBound)]
                targetStat = targetStat[~np.isnan(targetStat)]
                m = np.mean(targetStat)
                s = np.std(targetStat)

                if s == 0:
                    score.append(self.shiftedSigmoid(0))
                    continue
                yScore = self.shiftedSigmoid((yValue-m)/s)
                score.append(yScore)
            else:
                score.append(self.shiftedSigmoid(0))
            means.append(m)
            sds.append(s)

        means = np.array(means[170:-170])
        sds = np.array(sds[170:-170])
        plt.plot(target['x'][170:-170], means, 'b', label = "mean")
        plt.plot(target['x'][170:-170], means - sds, 'r', label = "1SD above")
        plt.plot(target['x'][170:-170], means + sds, 'r', label = "1SD below")
        plt.xlabel(x.name)
        plt.ylabel(y.name)
        # plt.xscale("log")
        plt.title("Distribution of mean and sd according to {}".format(x.name))
        plt.savefig("./data/pic/rankingStat/{} VS {}.png".format(x.name, y.name))
        plt.legend()
        plt.clf()

        target['y'] = score

        mx = statistics.mean(target['x'].dropna())
        sx = statistics.stdev(target['x'].dropna())
        target['x'] = (target['x']-mx)/sx
        target['x'] = target['x'].apply(self.shiftedSigmoid)

        target["mergeResult"] = target.apply(self.harmonicMean, axis = 1)
        orgTarget = orgTarget.merge(target, how = "left" ,on = "brandID")
        return orgTarget

    def ranking(self):
        """
        for each attri, get the merged score from rankAndDecorrelationAttri
        engagement is from the harmonic mean (AND) of likes, share and fb_avg_comments_num
        the fb/ig_score is from the merged score from rankAndDecorrelationAttri of followers and Engagement
        the final score is the harmonic mean (OR) of fb_score and ig_score

        the result is save to the "./data/brandsWithRanking.csv"
        """
        scoredBrands = self.scoredBrands
        balancedScore = pd.DataFrame(index = scoredBrands.index)
        for attri in self.neededAttriGrowth:
            print(attri)
            target = scoredBrands[attri]
            plt.hist(target[target<target.quantile(0.95)], bins = 200)
            plt.title("{} distribution".format(attri))
            plt.savefig("./data/pic/checkRanking/{}_distribution.png".format(attri))
            plt.clf()
            if (len(scoredBrands[attri].dropna() != 0 )):
                mergeResultDF = self.rankAndDecorrelationAttri(scoredBrands[attri], scoredBrands[attri+"_growth"])
                mergeResultDF.columns = ["{}".format(attri), "{}_growth".format(attri), "{}_merged".format(attri)]
                balancedScore = balancedScore.merge(mergeResultDF, how = 'left', on = "brandID")
            else:
                balancedScore[attri] = np.nan

        balancedScore["fb_engagement"] = (balancedScore.loc[:,["fb_avg_likes_num_merged", "fb_avg_shares_num_merged", "fb_avg_comments_num_merged"]]
                                            .apply(lambda row : self.harmonicMean(row, weight = [0.6,0.2,0.2]), axis = 1))
        balancedScore["ig_engagement"] = (balancedScore.loc[:,["ig_avg_likes_num_merged", "ig_avg_comments_num_merged"]]
                                            .apply(lambda row : self.harmonicMean(row, weight = [0.75,0.25]), axis = 1))

        balancedScore["fb_score"] = self.rankAndDecorrelationAttri(balancedScore["fb_followers_merged"], balancedScore["fb_engagement"])["mergeResult"]
        balancedScore["ig_score"] = self.rankAndDecorrelationAttri(balancedScore["ig_followers_merged"], balancedScore["ig_engagement"])["mergeResult"]

        print("balancedScore", balancedScore)
        balancedScore["final_score"] = balancedScore.loc[:,["fb_score", "ig_score"]].apply(lambda row : self.harmonicMean(row, p = 2), axis = 1)
        dfB = pd.read_csv("./data/brandsWithScoring.csv").set_index('brandID', drop = False)
        print("dfB \n", dfB)
        self.rankedBrands = balancedScore.combine_first(dfB)
        print("combinded \n",self.rankedBrands)
        print(self.rankedBrands.sort_values( "final_score", ascending = False).head())
        self.rankedBrands = self.rankedBrands.drop(['ig_posts', 'ig_posts_num', 'ig_followings', 'fb_posts_num'], axis = 1)
        self.rankedBrands.to_csv('./data/brandsWithRanking.csv', header = True, index = False)

    def plotScatter(self,attri,ground):
        """
        helper function for vsGroundPlot
        """
        q = 0.95
        target = self.scoredBrands[[ground, attri]]
        qA1 = target[ground].quantile(1-q)
        qA2 = target[ground].quantile(q)
        qB1 = target[attri].quantile(1-q)
        qB2 = target[attri].quantile(q)
        target = target[target[ground].between(qA1,qA2) & target[attri].between(qB1,qB2) ]
        x = target[ground]
        y = target[attri]
        #self.curveFitting(x,y)
        plt.scatter(x,y)
        plt.ylabel(attri)
        plt.xlabel(ground)
        plt.title("{} vs {}".format(attri, ground))
        plt.savefig("./data/pic/explore/{}vs{}.png".format(attri,ground))
        plt.clf()
        print("{} \t {} \t {}".format(ground, attri, np.corrcoef(x,y)[0][1]))

    def VSGroundPlot(self):
        """
        for visualizing the relationship between different grounds and attri
        """
        for attri in self.neededAttri_FB:
            ground = "fb_followers"
            if attri == ground:
                continue
            self.plotScatter(attri,ground)

        for attri in self.neededAttri_IG:
            ground = "ig_followers"
            if attri == ground:
                continue
            self.plotScatter(attri,ground)

        for attri in self.neededAttriGrowth:
            ground = attri + "_growth"
            self.plotScatter(ground,attri)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vsGround',action='store_true', help='compare the variables with different grounds')
    parser.add_argument('--basicScoring',action='store_true', help='calculate basic predicted scoring of the brand')
    parser.add_argument('--ranking',action='store_true', help='clear the result CSV')
    parser.add_argument('--clearFile',action='store_true', help='clear the result CSV')
    parser.add_argument('--checkDistribution',action='store_true', help='clear the result CSV')
    parser.add_argument('--checkTrend',action='store_true', help='clear the result CSV')
    parser.add_argument('--interpretRanking',action='store_true', help='clear the result CSV')
    args = parser.parse_args()

    ex = scoringAndRanking()

    if args.clearFile:
        try:
            os.remove('./data/brandsWithScoring.csv')
        except:
            pass
    ex.setUpBrands(args.clearFile)
    if args.checkTrend:
        while(1):
            for i, attri in enumerate(ex.neededAttri):
                print(i, attri)
            brandID = input("brandID\t")
            attriIndex = input("attri\t")
            ex.plotTrend(int(brandID), ex.neededAttri[int(attriIndex)])
    if args.checkDistribution:
        ex.checkDistributionOfRecent()
    if args.ranking:
        ex.ranking()
    if args.interpretRanking:
        ex.interpretRanking()
    if args.basicScoring:
        with suppress_stdout_stderr():
            ex.basicScoring()
    if args.vsGround:
        ex.VSGroundPlot()

if __name__ == "__main__":
    main()
