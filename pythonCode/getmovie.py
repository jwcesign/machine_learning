# -*- coding:utf-8 -*-
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from bs4 import BeautifulSoup
import time
import re
from numpy import *

#global data
dcap = dict(DesiredCapabilities.PHANTOMJS)
dcap["phantomjs.page.settings.userAgent"] = (
	    "Mozilla/5.0 (X11; Linux x86_64; rv:45.0) Gecko/20100101 Firefox/45.0"
	    )
driver = webdriver.PhantomJS(desired_capabilities=dcap)
driver.set_window_size(1920, 1080)

def getList(url):
	driver.get(url)

	#find the button
	more = driver.find_elements_by_class_name("more")
	
	#click the button more,while can't find, it will end
	try:
		while True:
			more[0].click()
			more = driver.find_elements_by_class_name("more")
	except Exception:
		source=driver.page_source.encode('utf-8')


	fs = open('movie.txt','w')


	com = re.compile(r'<a class="item"[^<]*?>')
	urlFirst = str(re.findall(com,source))
	com = re.compile(r'href=".*?"')
	urlFinal = re.findall(com,urlFirst)

	print len(urlFinal)

	#save the urls
	for i in urlFinal:
		i = i.replace('href="','').replace('"','')
		fs.write(i+'\n')

	fs.close()
	return urlFinal

def getData(url,fs):
	url = url.replace('href="','').replace('"','')
	driver.get(url)
	source = driver.page_source.encode('utf-8')

	##get title
	reTitle = re.compile(r'itemreviewed">.*?</span>')
	title = re.findall(reTitle,source)
	reTitle = re.compile(r'[^>]+<')
	title = re.findall(reTitle,title[0])
	title = title[0].replace('<','')
	#print title

	##get score
	reScore = re.compile(r'average">.*?</strong>')
	score = re.findall(reScore,source)
	reScore = re.compile(r'[^>]+<')
	score = re.findall(reScore,score[0])
	score = score[0].replace('<','')
	#print score

	##get comment
	reComment = re.compile(r'<p class="">[^<]+')



	comment = re.findall(reComment,source)
	#print comment[0]
	#print comment[1]

	if len(comment)>=2:
		comment = str(comment[0]+comment[1])
		comment = comment.replace('<p class="">','')
		comment = comment.split('\n')
		comment[0] = comment[0].replace(' ','')
		comment[1] = comment[1].replace(' ','')
	else:
		comment = 'Null'
	
	#driver.get_screenshot_as_file('show.png')

 
	fs.write(title+'---'+score+'---'+comment[0]+'---'+comment[1]+'\n')

def startScan():
	url = 'https://movie.douban.com/'
	urls = getList(url)
	file = open('data.txt','w')
	indexNow = 1
	total = float(len(urls))
	for i in urls:
		getData(i,file)
		print '>>>'+str(indexNow*100/total)+'%'
		indexNow = indexNow+1
	file.close()


##the machine learning part

#get data set
def loadDataSet(addr):
	file = open(addr)
	line = file.readline()
	data = line.split('---')
	dataG = []
	dataS = []
	if len(data) >=4:
		comment = str(data[2]+data[3]).decode('utf8')
		comment = re.sub(r'[“”#，。～ ？?a-zA-z.:：*&%$#@$￥+=-——_0-9.\n]*'.decode('utf8'),'',comment)
		score = str(data[1])
	else:
		comment = str(data[2]).decode('utf8')
		comment = re.sub(r'[“”#，。～ ？?a-zA-z.:：*&%$#@$￥+=-——_\n]*'.decode('utf8'),'',comment)
		score = str(data[1])
	
	while line:
		dataG.extend([list(comment)])
		if float(data[1])>=7.5:
			dataS.append(1)
		else:
			dataS.append(0)
		line = file.readline()
		data = line.split('---')
		if len(data)>=3:
			if len(data) >=4:
				comment = str(data[2]+data[3]).decode('utf8')
				comment = re.sub(r'[“”#，。～ ？?a-zA-z.:：*&%$#@$￥+=-——_0-9.\n]*'.decode('utf8'),'',comment)
				score = str(data[1])
			else:
				comment = str(data[2]).decode('utf8')
				comment = re.sub(r'[“”#，。～ ？?a-zA-z.:：*&%$#@$￥+=-——_\n]*'.decode('utf8'),'',comment)
				score = str(data[1])
		
	return dataG,dataS

##create data dic
def createVocabList(dataSet):
	vocabSet = set([])
	for doc in dataSet:
		vocabSet = vocabSet | set(doc)
	return list(vocabSet)

##get the vertex of vocablist
def setOfWords2Vec(vocabList,inputSet):
	returnVec = [0]*len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] += 1
		else:
			print 'The word ',word,' is not in my vocabulary'
	return returnVec

##training function
def trainNB0(trainMatrix,trainCategory):
	numTrainDocs = len(trainMatrix)
	numWords = len(trainMatrix[0])
	pAbusive = sum(trainCategory)/float(numTrainDocs)
	#***
	p0Num = ones(numWords)
	p1Num = ones(numWords)
	p0Denom = 2.0
	p1Denom = 2.0
	#***
	for i in range(numTrainDocs):
		if trainCategory[i] == 1:
			p1Num += trainMatrix[i]
			p1Denom += sum(trainMatrix[i])
		else:
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])
	#***
	p1Vect = log(p1Num/p1Denom)
	p0Vect = log(p0Num/p0Denom)
	return p0Vect,p1Vect,pAbusive
	#***

def classifyNB(vec2Classify, p0Vec,p1Vec,pClass1):
	p1 = sum(vec2Classify * p1Vec) + log(pClass1)
	p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
	if p1 > p0:
		return 'Good movie, you should see it'
	else:
		return 'Bad movie, do not see it'

# def testingNB():
# 	###start###
# 	listOPosts,listClasses = loadDataSet('data.txt')
# 	myVocabList = createVocabList(listOPosts)
# 	trainMat = []
# 	for postinDoc in listOPosts:
# 		trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
# 	p0v,p1v,pAb = trainNB0(array(trainMat),array(listClasses))
# 	testEntry = ['好','非','情']
# 	thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
# 	print thisDoc,'---',p0v,'---',p1v,'---',pAb
# 	print testEntry,'classified as: ',classifyNB(thisDoc,p0v,p1v,pAb)
# 	testEntry = ['圾','不']
# 	thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
# 	print testEntry,' classified as: ',classifyNB(thisDoc,p0v,p1v,pAb)


def getDataSet():
	startScan()
	driver.close

def checkP(str):
	str = re.sub(r'[“”#，。～ ？?a-zA-z.:：*&%$#@$￥+=-——_0-9.\n]*'.decode('utf8'),'',str.decode('utf8'))
	listOPosts,listClasses = loadDataSet('data.txt')
	myVocabList = createVocabList(listOPosts)
	trainMat = []
	for postinDoc in listOPosts:
		trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
	p0v,p1v,pAb = trainNB0(array(trainMat),array(listClasses))
	testEntry = list(str)
	thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
	print 'The comment is classified as: ',classifyNB(thisDoc,p0v,p1v,pAb)
