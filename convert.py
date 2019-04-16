import json

output = {}
rounding_digits = 4
with open('./input/stockPriceRaw.json') as json_file:
    stocks = json.load(json_file)
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

    with open('./input/stockPrices.json', 'w') as outfile:
        json.dump(output, outfile)