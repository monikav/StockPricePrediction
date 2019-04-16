import sys
import json
import os



stocks = {}
rounding_digits = 4
path = './csv'
for filename in os.listdir(path):
	if filename == '.DS_Store':
		continue
	with open('./csv/'+ filename) as f:

		data = {}
		i = 0 
		for line in f:
			i = i + 1
			if i == 1 :
				continue
			date,openval,high,low,close,adjClose,volume = line.split(",")
			
			year,month,day = date.split("-")
			date = year+month+day
			data[date] = {}	
			data[date]["open"] = openval
			data[date]["high"]= high
			data[date]["low"]= low
			data[date]["close"] = close
			data[date]["volume"]= volume
			data[date]["adjClose"] = adjClose

	stocks[filename.rstrip(".csv")] = data		

output = {}
prev = 0
for stock in stocks:
    output[stock] = {}
    j = 0;
    for date in sorted(stocks[stock].items(), key=lambda s: s[0]):
        #output[stock][date[0]] = date[1]
        if j == 0:
            prev = date[1]['close']
            #['change']
        output[stock][date[0]] = round(float(date[1]['close']) - float(prev), rounding_digits)
        j = j+1
        prev = date[1]['close']

with open('input/stockPrices.json', 'w') as outfile:
	json.dump(output, outfile)




