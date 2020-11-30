import warnings
warnings.filterwarnings('ignore')
import sys
import os
import datetime
import numpy as np
import pandas as pd
import math
from os import listdir
from os.path import isfile, join
import re

class DataPreporation:
    def __init__(self):
        self.collectAllData()
        self.removeDuplicatedBrands()
        self.filterDataByCount()
        self.exportResult()
        pass

    def getUsername(self, link):
        fbSearchRegex = '(?<=.com/)\w+'
        fb_url = 'https://facebook.com/accounts/login/?next...'

        if not(isinstance(link,str)):
            return math.nan
        m = re.search(fbSearchRegex, link)
        if bool(m) :
            if (m.group(0) != 'accounts') :
                return m.group(0)
        return math.nan

    def collectAllData(self):
        dataPath = "./data/socialstat/"

        allName = [f for f in listdir(dataPath) if isfile(join(dataPath, f))]
        allDFList = [pd.read_csv(dataPath + f) for f in listdir(dataPath) if isfile(join(dataPath, f))]
        allData = pd.concat(allDFList, keys=allName)

        neededColumns = ['id', 'brand', 'cat_1', 'cat_2', 'cat_3', 'cat_4', 'country',
               'fb_likes',
               'fb_followers',
               'fb_posts_num', 'fb_avg_comments_num',
               'fb_avg_likes_num', 'fb_avg_shares_num',
               'ig_followings',
               'ig_followers',
               'ig_posts','ig_hashtags_num',
               'ig_posts_num', 'ig_avg_comments_num',
               'ig_avg_likes_num', 'fb_username', 'ig_username',
               'fb_url', 'ig_url', 'created',
               'modified']
        allNeededData = allData.loc[:,neededColumns]
        allNeededData.loc[allNeededData.fb_url == "https://facebook.com/", 'fb_url'] = math.nan
        allNeededData.loc[allNeededData.ig_url == "https://instagram.com/", 'ig_url'] = math.nan

        allNeededData['fb_username'] = allNeededData['fb_url'].apply(self.getUsername)
        allNeededData['ig_username'] = allNeededData['ig_url'].apply(self.getUsername)

        self.allNeededData = allNeededData

    def findUniqueCorrespondant (self, groupByCol, countCol):
        s = self.allNeededData.groupby(groupByCol)[countCol].nunique()
        sOne2One = s[s <= 1]
        sOne2Multi = s[s > 1]
        return sOne2One

    #unique correspondant
    def checkUC(self, fb_username , ig_username):

        isString1 = isinstance(fb_username, str)
        isString2 = isinstance(ig_username, str)

        if (not (isString1 | isString2)):
            return False

        flag1 = (True if not isString1 else fb_username in self.One2OneFB2IG)
        flag2 = (True if not isString2 else ig_username in self.One2OneIG2FB)
        return flag1 & flag2

    def fetchDFByUserName(self, fb_url , ig_url):
        isString_FB = isinstance(fb_url, str)
        isString_IG = isinstance(ig_url, str)

        if (isString_FB & isString_IG):
            result = self.allBrands[(self.allBrands['fb_username'] == fb_url) & (self.allBrands['ig_username'] == ig_url) ]
        elif isString_FB:
            result = self.allBrands[self.allBrands['fb_username']== fb_url]
        else :
            result = self.allBrands[self.allBrands['ig_username']== ig_url]
        return result

    def fetchDFForAValue(self, x, fieldNames):
        df = self.fetchDFByUserName(x.fb_username , x.ig_username)
        for fieldName in fieldNames:
            x[fieldName] = df[fieldName].dropna().iloc[0] if (df[fieldName].dropna().size != 0) else math.nan
        return x

    def removeDuplicatedBrands(self):
        self.One2OneFB2IG = self.findUniqueCorrespondant ('fb_username', 'ig_username').index
        self.One2OneIG2FB = self.findUniqueCorrespondant ('ig_username', 'fb_username').index

        self.allNeededData['UC_check'] = self.allNeededData.apply(lambda x: self.checkUC(x['fb_username'], x['ig_username']), axis=1)
        allNeededDataCleanName = self.allNeededData[ self.allNeededData['UC_check'] == True]

        self.allBrands = allNeededDataCleanName.loc[:,["fb_username", "ig_username","brand",'cat_1', 'cat_2', 'cat_3', 'cat_4', 'country']].drop_duplicates()

        uniqueBrands = allNeededDataCleanName.loc[:,["fb_username", "ig_username"]].drop_duplicates()

        fieldNames = ["fb_username", "ig_username"]
        uniqueBrands = uniqueBrands.apply(lambda x : self.fetchDFForAValue(x, fieldNames), axis = 1)
        uniqueBrands = uniqueBrands.drop_duplicates()

        fieldNames = ['brand','cat_1', 'cat_2', 'cat_3', 'cat_4', 'country']
        uniqueBrands = uniqueBrands.apply(lambda x : self.fetchDFForAValue(x, fieldNames), axis = 1)

        uniqueBrands = uniqueBrands.reset_index()
        uniqueBrands["brandID"] = uniqueBrands.index + 1
        uniqueBrands = uniqueBrands.drop(['level_0', 'level_1'], axis = 1)

        self.uniqueBrands = uniqueBrands

        allNeededDataCleanName = allNeededDataCleanName.reset_index().rename(columns={"level_0": "orgFile", "level_1": "orgFileIndex"})
        allNeededDataCleanName["timeInstanceID"] = allNeededDataCleanName.index + 1
        cleanTimeInstance = allNeededDataCleanName.drop(['brand', 'cat_1','cat_2','cat_3','cat_4','country' ], axis = 1)
        cleanTimeInstance.created = pd.to_datetime(cleanTimeInstance.created)

        cleanTimeInstance_withFB = cleanTimeInstance[cleanTimeInstance.apply(lambda x : isinstance(x['fb_username'],str), axis = 1)]
        cleanTimeInstance_withoutFB = cleanTimeInstance[cleanTimeInstance.apply(lambda x : not isinstance(x['fb_username'],str), axis = 1)]

        fullTimeInstance_withFB = pd.merge(cleanTimeInstance_withFB.drop(['ig_username'], axis = 1), uniqueBrands.dropna(subset=['fb_username']), on="fb_username", how = 'left')
        fullTimeInstance_withoutFB = pd.merge(cleanTimeInstance_withoutFB.drop(['fb_username'], axis = 1), uniqueBrands.dropna(subset=['ig_username']), on="ig_username", how = 'left')

        fullTimesInsatance = fullTimeInstance_withFB.append(fullTimeInstance_withoutFB)
        self.fullTimesInsatance = fullTimesInsatance.groupby(['brandID', 'created']).last().reset_index()

    def countFiltering (self,x):
        try:
            return (self.countInPercentage[x['brandID']] > 0.6) & (self.dayCountByBrandID[x['brandID']] > 30)
        except:
            return False

    def filterDataByCount(self):

        fullTimesInsatance = self.fullTimesInsatance.copy()
        fullTimesInsatance = fullTimesInsatance.groupby(['brandID', 'created']).last().reset_index()
        brandsObservationsCount = fullTimesInsatance.groupby('brandID').count()

        dayCountByBrandID = fullTimesInsatance[['created',"brandID"]].groupby('brandID')
        dayCountByBrandID = dayCountByBrandID.max() - dayCountByBrandID.min()
        dayCountByBrandID = dayCountByBrandID['created'].dt.days.astype('int16') +1
        self.dayCountByBrandID = dayCountByBrandID.fillna(1)

        self.countInPercentage = brandsObservationsCount['timeInstanceID']/dayCountByBrandID
        self.fullTimesInsatance_filtered = fullTimesInsatance[fullTimesInsatance.apply(self.countFiltering , axis=1)]

    def exportResult(self):
        self.uniqueBrands.to_csv('./data/uniqueBrands.csv')
        self.fullTimesInsatance.to_csv('./data/fullTimesInsatance.csv')
        self.fullTimesInsatance_filtered.to_csv('./data/fullTimesInsatance_filtered.csv')

data = DataPreporation()
