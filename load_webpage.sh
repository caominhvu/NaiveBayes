#!/bin/bash

cd ../WebCrawler

scrapy crawl webpage --set FEED_URI=page.json --set FEED_FORMAT=json -a url="http://vnexpress.net/tin-tuc/oto-xe-may/bmw-m-moi-lo-anh-truoc-thoi-diem-ra-mat-2922448.html"

mv *.json ../NaiveBayes/page

cd ../NaiveBayes