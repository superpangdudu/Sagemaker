
import scrapy

class MySpider(scrapy.Spider):
    name = 'myspider'
    start_urls = [
        'http://163.com'
    ]


