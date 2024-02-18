# A web scraper for nepali news sites - Ekantipur

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
import codecs

# Path for chrome driver
# check if chrome driver has expired
chrome_driver = "D:/Development/chromedriver.exe"

DCNEPAL_WEBSITES = {
    #   "अर्थ / वाणिज्य": "https://www.dcnepal.com/category/business/",
    #   "विश्व": "https://www.dcnepal.com/category/international/",
    #   "विचार": "",
    "राजनीति": "https://www.dcnepal.com/category/political/",
    # "खेलकुद": "https://www.dcnepal.com/category/sports/",
    #   "स्वास्थ्य": "https://www.dcnepal.com/category/health-lifestyle/",
    #   "विज्ञान र प्रविधि": "https://www.dcnepal.com/category/technology/",
    # "समाज": "https://www.dcnepal.com/category/society/",
    #   "साहित्य/विविध": "",
    #   "अन्य": "",
    # "देश/प्रदेश": "https://www.dcnepal.com/category/desh-pradesh/",
    #   "मनोरञ्जन": "https://www.dcnepal.com/category/entertainment-news/",
    #   "कला": "",
    #   "प्रवास": "",
}

options = webdriver.ChromeOptions()
options.add_experimental_option("excludeSwitches", ["enable-logging"])

service = Service(chrome_driver)
driver = webdriver.Chrome(service=service, options=options)

# TOTAL_NEWS_TO_SELECT = 4020

for key, value in DCNEPAL_WEBSITES.items():
    driver.get(value)
    time.sleep(2)

    total_news = 0

    # Get the news list from the each page until 150 items are selected
    for page in range(31, 40):
        news_list = []
        new_page = value
        if page > 1:
            new_page = value + f"/page/{page}/"
            driver.get(new_page)
            time.sleep(2)

        # banner_news = driver.find_element(By.CSS_SELECTOR, 'div.category__container div.grid div.column-3 div.row--news')
        # news_list.append(banner_news)
        category_news = driver.find_elements(
            By.CSS_SELECTOR,
            "div.category__container div.grid div.column-3 div.row--news",
        )
        # if page == 45:
        #     category_news = category_news[:22]
        news_list += category_news

        # total_news += len(news_list)

        # if total_news > TOTAL_NEWS_TO_SELECT:
        #     break

        news_cover_links = []
        news_titles = []
        for news_cover in news_list:
            news_link = news_cover.find_element(
                By.CSS_SELECTOR, "h3.news__title--small a.title"
            )
            news_titles.append(news_link.text)
            news_cover_links.append(news_link.get_attribute("href"))

        with codecs.open("dcnepal_news_links.csv", "a", "utf-8") as file:
            file.write("\n".join(news_cover_links))
            file.write("\n")

        with codecs.open("dcnepal_news_titles.csv", "a", "utf-8") as file:
            file.write("\n".join(news_titles))
            file.write("\n")
        continue
        # exit()
