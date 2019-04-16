#!/usr/bin/python
import urllib3
import re
import sys
import time
from datetime import timedelta, date
import numpy as np
import csv
import os
import json

from bs4 import BeautifulSoup


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)



def dateGenerator(date1, date2): # generate list of dates.
    dateList = []
    for dt in daterange(date1,date2):
        dateList.append(dt.strftime("%Y%m%d"))
    return dateList

def daterange(date1, date2):
    for n in range(int ((date2 - date1).days)+1):
        yield date1 + timedelta(n)


def contents(symbol, name, line, dateList, exchange):
    suffix = {'AMEX': '.A', 'NASDAQ': '.O', 'NYSE': '.N'}
    has_Content = 0
    repeat_number = 4
    newsUrl = "http://www.reuters.com/finance/stocks/company-news/" + symbol + suffix[exchange]
    http = urllib3.PoolManager()
    
    for x in range(repeat_number): # repeat in case of failure
        try:
            time.sleep(np.random.poisson(3))
            result = http.request('GET', newsUrl)
            soup = BeautifulSoup(result.data, "lxml")
            has_Content = len(soup.find_all("div", {'class': ['topStory', 'feature']}))
            break
        except Exception as e:
            print(e)
            continue


    failed_symbol = open('./input/news_failed_tickers.csv', 'a+')
    if has_Content > 0:
        empty_streak = 0
        for timestamp in dateList:
            news_available = downloadContent(symbol, line, newsUrl, timestamp)
            if news_available: 
                empty_streak = 0 # if get news, reset empty_streak as 0
            else: 
                empty_streak += 1
            if empty_streak > has_Content * 5 + 20: # 2 NEWS: wait 30 days and stop,
                break 
            if empty_streak > 0 and empty_streak % 20 == 0: 
                print("%s has no news for %d days" % (symbol, empty_streak))
                failed_symbol.write(symbol + ',' + timestamp + ',' + 'LOW\n')
    else:
        print("%s does not have news" % (symbol))
        today = date.today().strftime("%Y%m%d")
        failed_symbol.write(symbol + ',' + today + ',' + 'LOWEST\n')
    failed_symbol.close()



def parseHtml(soup, line, symbol, timestamp):
    body = soup.find_all("div", {'class': ['topStory', 'feature']})
    if len(body) == 0: return 0
    news_collected = open('./input/news_reuters.csv', 'a+',encoding = 'utf-8')
    for i in range(len(body)):
        heading = body[i].h2.get_text().replace(",", " ").replace("\n", " ")
        para = body[i].p.get_text().replace(",", " ").replace("\n", " ")

        if i == 0 and len(soup.find_all("div", class_="topStory")) > 0: news_type = 'topStory'
        else: news_type = 'normal'

        print(symbol, timestamp, heading, news_type)
        news_collected.write(','.join([symbol, line[1], timestamp, heading, para, news_type])+'\n')

    news_collected.close()
    return 1


def downloadContent( symbol, line, newsUrl, timestamp):
     # change YMD to MDY to match reuters date format
    new_time = timestamp[4:] + timestamp[:4]
    http = urllib3.PoolManager()
    repeat_number = 3 # repeat downloading in case of  error
    for _ in range(repeat_number):
        try:
            time.sleep(np.random.poisson(3))
            result = http.request('GET', newsUrl + "?date=" + new_time)
            soup = BeautifulSoup(result.data, "lxml")
            news_available = parseHtml(soup, line, symbol, timestamp)
            if news_available: 
                return 1 # return if we get the news
            break 
        except Exception as e: # repeat if http error occurs
            print(e)
            continue
    return 0


def crawl_News_Reuters():
    finput = open('./input/symbolsList.csv')
    start_dt = date(2017, 1, 1)
    end_dt = date(2019, 1, 3)
   
    dateList = dateGenerator(start_dt, end_dt)
    for line in finput:
        line = line.strip().split(',')
        symbol, name, exchange = line
        print("%s - %s - %s" % (symbol, name, exchange))
        contents(symbol, name, line, dateList, exchange)

def main():
    crawl_News_Reuters()

if __name__ == "__main__":
    main()
