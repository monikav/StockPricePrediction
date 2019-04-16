import sys
import json
import os



ticker = {}
path = './csv'
for filename in os.listdir(path):
	with open('./csv/'+ filename) as f:
		data = {}
		data["open"] = {}
		data["high"] = {}
		data["low"] = {}
		data["close"] = {}
		data["adjClose"] = {}
		data["volume"] = {}
		i = 0 
		for line in f:
			i = i + 1
			if i == 1 :
				continue
			date,openval,high,low,close,adjClose,volume = line.split(",")
			data["open"][date] = openval
			data["high"][date]= high
			data["low"][date]= low
			data["close"][date]= close
			data["volume"][date] = volume
			data["adjClose"][date] = adjClose

	ticker[filename.rstrip(".csv")] = data		


with open('./input/stockPriceRaw.json', 'w') as outfile:
	json.dump(ticker, outfile)




