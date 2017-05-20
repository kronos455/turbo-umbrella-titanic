#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 01:41:48 2017

@author: bmitchell
"""

import re	
import csv 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from datetime import datetime
from sklearn.cross_validation import cross_val_score

# Add time along with the log
def log(string):
        print(str(datetime.now()) + " " + string)


# Convert the gender
def convertGender(gender):
	if gender == 'female':
		gender = 0
	if gender == 'male':
		gender = 1
	return gender

# Convert the embarked field
def convertEmbarked(embarked):
	if embarked == 'C':
		embarked = 0
	if embarked == 'Q':
		embarked = 1
	if embarked == 'S':
		embarked = 2
	else:
		embarked = '2'
	return embarked

# return title
def getTitle(name):
	for word in name.split():
		if word.endswith('.'):
			title=word
			break
	return title

# convert title to hash 
# TODO need to improve
def getTitleHash(title,gender):
	has = ord(title[0]) + len(title) + int(gender)
	return has

# returns one if the passenger had a family
def getFamily(sibsp,parch):
	if int(sibsp) + int(parch) > 0:
		family = 1
	else:
		family = 0
	return family

# Pull out the dept from the ticket number
def getTicketCode(ticket):
    deptName = re.sub(r"$\d+\W+|\b\d+\b|\W+\d+$", "", ticket)
    if len(deptName) == 0:
        deptName = 'none'
    deptCode = ord(deptName[0]) + len(deptName)
    return deptCode

if __name__ == '__main__':	

	log("Reading Train Data")
	
	train = csv.reader(open('train.csv','rb'))
	#header = train.next()
	
	######READING TRAIN DATA################	
	train_data=[]
	for row in train:
	        train_data.append(row)
	
	train_data = np.array(train_data)
	
	log("DONE Reading Train Data")
	
	log("Preprocessing Train Data")
	# replace categorical attributes
	for row in train_data:
		
		row[4] = convertGender(row[4])
		title = getTitle(row[3])
		row[3] = getTitleHash(title,row[4])
		row[6] = getFamily(row[6],row[7])
		row[8] = getTicketCode(row[8])
		row[11] = convertEmbarked(row[11])
		

	features = train_data[0::,[2,3,4,6,8,11]]
	result = train_data[0::,1]
	log("DONE Preprocessing Train Data")

	log("Fitting Train Data")
	adaBoost = AdaBoostClassifier(RandomForestClassifier(n_estimators = 1000),
                         algorithm="SAMME",
                         n_estimators=200)
	
	adaBoost = adaBoost.fit(features,result)
	log("DONE Fitting Train Data")

	log("Calculating score")
	scores = cross_val_score(adaBoost, features, result)
	log("Training score")
	print(scores.mean())
	log("DONE Calculating score")
	

	######READING TEST DATA################	
	log("Reading Test Data")
	test = csv.reader(open('test.csv','rb'))
	#header = test.next()
	
	test_data=[]
	for row in test:
	        test_data.append(row)
	test_data = np.array(test_data)
	log("DONE Reading Test Data")
	
	# replace categorical attributes
	log("Preprocessing Test Data")
	for row in test_data:
		
		row[3] = convertGender(row[3])
		title = getTitle(row[2])
		row[2] = getTitleHash(title,row[3])
		row[5] = getFamily(row[5],row[6])
		row[7] = getTicketCode(row[7])
		row[10] = convertEmbarked(row[10])
		

	features = test_data[0::,[1,2,3,5,7,10]]
	log("DONE Preprocessing Test Data")
	
	log("Predicting Test Data")
	Output = adaBoost.predict(features)
	
	np.savetxt("adaBoostRandomForest.csv",Output,delimiter=",",fmt="%s")	
	log("DONE Predicting Test Data")