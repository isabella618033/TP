#!/bin/bash

cd ./data
find . -type d ! -name "socialstat" -delete
cd ..
mkdir ./data/filledData/
mkdir ./data/pic/
mkdir ./data/pic/explore/
mkdir ./data/pic/explore/allFBFollowers/
mkdir ./data/pic/explore/suspiciousFBFollowers/
mkdir ./data/pic/rankingStat/
mkdir ./data/pic/checkRanking/
/
