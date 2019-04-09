from bs4 import BeautifulSoup
import datetime
import pickle
import requests

companyList = "CompanyList.txt"
stocks = []

with open(companyList) as f:
    stocks = f.readlines()
for index in range(len(stocks)):
    stocks[index] = stocks[index].rstrip()

def getNewsForDate(date):
    newsFile = open('data/news/'+ date.strftime("%Y-%m-%d") + ".csv", "w")
    for i in range(len(stocks)):
        request = 'http://www.reuters.com/finance/stocks/companyNews?symbol='+  stocks[i] + '&date=' + format(date.month, '02d') + format(date.day, '02d') + str(date.year)
        response = requests.get(request)
        parser = BeautifulSoup(response.text, "html.parser")
        divisions = parser.findAll('div', {'class':'feature'})
        if(len(divisions) == 0):
            continue
        data = u''
        for div in divisions:
            data = data.join(div.findAll(text=True))
        newsFile.write(stocks[i] + ',' + data.replace('\n', ' '))
        newsFile.write('\n')
    newsFile.close()
	
date = datetime.date(2016,10,11) + datetime.timedelta(days=1)
endDate = datetime.date(2017,10,11)
while date<=endDate:
    getNewsForDate(date)
    date += datetime.timedelta(days = 1) 





