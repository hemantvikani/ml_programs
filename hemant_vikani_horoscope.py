# -*- coding: utf-8 -*-
"""
Created on Thu May 31 22:31:21 2018

@author: heman
"""


import urllib2
from bs4 import BeautifulSoup






sign=raw_input("enter your sign")
if sign is "aries":
    wiki="https://www.horoscope.com/us/horoscopes/general/horoscope-general-daily-today.aspx?sign=1"
    page = urllib2.urlopen(wiki)
    soup = BeautifulSoup(page)
    soup = soup.find(class_="horoscope-content")
    soup = soup.find('p')
    print soup
elif sign=="taurus":
    wiki="https://www.horoscope.com/us/horoscopes/general/horoscope-general-daily-today.aspx?sign=2"
    page = urllib2.urlopen(wiki)
    soup = BeautifulSoup(page)
    soup = soup.find(class_="horoscope-content")
    soup = soup.find('p')
    print soup
elif sign=="gemini":
    wiki="https://www.horoscope.com/us/horoscopes/general/horoscope-general-daily-today.aspx?sign=3"
    page = urllib2.urlopen(wiki)
    soup = BeautifulSoup(page)
    soup = soup.find(class_="horoscope-content")
    soup = soup.find('p')
    print soup
elif sign=="cancer":
    wiki="https://www.horoscope.com/us/horoscopes/general/horoscope-general-daily-today.aspx?sign=4"
    page = urllib2.urlopen(wiki)
    soup = BeautifulSoup(page)
    soup = soup.find(class_="horoscope-content")
    soup = soup.find('p')
    print soup
elif sign=="leo":
    wiki="https://www.horoscope.com/us/horoscopes/general/horoscope-general-daily-today.aspx?sign=5"
    page = urllib2.urlopen(wiki)
    soup = BeautifulSoup(page)
    soup = soup.find(class_="horoscope-content")
    soup = soup.find('p')
    print soup
elif sign=="virgo":
    wiki="https://www.horoscope.com/us/horoscopes/general/horoscope-general-daily-today.aspx?sign=6"
    page = urllib2.urlopen(wiki)
    soup = BeautifulSoup(page)
    soup = soup.find(class_="horoscope-content")
    soup = soup.find('p')
    print soup
elif sign=="libra":
    wiki="https://www.horoscope.com/us/horoscopes/general/horoscope-general-daily-today.aspx?sign=7"
    page = urllib2.urlopen(wiki)
    soup = BeautifulSoup(page)
    soup = soup.find(class_="horoscope-content")
    soup = soup.find('p')
    print soup
elif sign=="scorpio":
    wiki="https://www.horoscope.com/us/horoscopes/general/horoscope-general-daily-today.aspx?sign=8"
    page = urllib2.urlopen(wiki)
    soup = BeautifulSoup(page)
    soup = soup.find(class_="horoscope-content")
    soup = soup.find('p')
    print soup
elif sign=="sagittarius":
    wiki="https://www.horoscope.com/us/horoscopes/general/horoscope-general-daily-today.aspx?sign=9"
    page = urllib2.urlopen(wiki)
    soup = BeautifulSoup(page)
    soup = soup.find(class_="horoscope-content")
    soup = soup.find('p')
    print soup
elif sign=="capricorn":
    wiki="https://www.horoscope.com/us/horoscopes/general/horoscope-general-daily-today.aspx?sign=10"
    page = urllib2.urlopen(wiki)
    soup = BeautifulSoup(page)
    soup = soup.find(class_="horoscope-content")
    soup = soup.find('p')
    print soup
elif sign=="aquarius":
    wiki="https://www.horoscope.com/us/horoscopes/general/horoscope-general-daily-today.aspx?sign=11"
    page = urllib2.urlopen(wiki)
    soup = BeautifulSoup(page)
    soup = soup.find(class_="horoscope-content")
    soup = soup.find('p')
    print soup
elif sign=="pisces":
    wiki="https://www.horoscope.com/us/horoscopes/general/horoscope-general-daily-today.aspx?sign=12"
    page = urllib2.urlopen(wiki)
    soup = BeautifulSoup(page)
    soup = soup.find(class_="horoscope-content")
    soup = soup.find('p')
    print soup
