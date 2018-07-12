# SpeechRecognition
## Dataset
data set is at Link: https://pan.baidu.com/s/1dlzGkABaMFH3e_qWJtNW9g passwd: xp8x
## Info
A simple speech recognition system which can is aimed at single word.  
## Process
1. find end point from the original sound file(zcr.py && volume.py)
2. get mfcc feature from the processed sound file( framenum * 39)
3. train model with SVM and mfcc data
4. use reliable model to recognise new sounds
