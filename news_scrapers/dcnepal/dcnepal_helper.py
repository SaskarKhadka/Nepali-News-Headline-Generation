from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
import codecs
import pandas as pd

# Path for chrome driver
# check if chrome driver has expired
chrome_driver = "D:/Development/chromedriver.exe"

df = pd.DataFrame(columns=["title", "news", "category", "published_date"])

options = webdriver.ChromeOptions()
options.add_experimental_option("excludeSwitches", ["enable-logging"])

service = Service(chrome_driver)
driver = webdriver.Chrome(service=service, options=options)

news_links = []
news_titles = []

with codecs.open("dcnepal_news_links.csv", "r", "utf-8") as file:
    news_links = file.readlines()

with codecs.open("dcnepal_news_titles.csv", "r", "utf-8") as file:
    news_titles = file.readlines()

DCNEPAL_WEBSITES = {
    #   "अर्थ / वाणिज्य": "https://www.dcnepal.com/category/business/",
    #   "विश्व": "https://www.dcnepal.com/category/international/",
    #   "विचार": "",
    # "खेलकुद": "https://www.dcnepal.com/category/sports/",
    "राजनीति": "https://www.dcnepal.com/category/political/",
    #   "स्वास्थ्य": "https://www.dcnepal.com/category/health-lifestyle/",
    #   "विज्ञान र प्रविधि": "https://www.dcnepal.com/category/technology/",
    # "समाज": "https://www.dcnepal.com/category/society/",
    # "देश/प्रदेश": "https://www.dcnepal.com/category/desh-pradesh/",
    #   "साहित्य/विविध": "",
    #   "अन्य": "",
    # "मनोरञ्जन": "https://www.dcnepal.com/category/entertainment-news/",
    #   "कला": "",
    #   "प्रवास": "",
}

for key, value in DCNEPAL_WEBSITES.items():
    for index, news_cover_link in enumerate(news_links):
        # news_title = news_cover.find_element(By.CSS_SELECTOR, 'div.teaser.offset h2 a')
        # title = news_title.text
        # driver.execute_script("arguments[0].scrollIntoView();", news_title)
        # driver.execute_script("arguments[0].click();", news_title)
        try:
            driver.get(news_cover_link)
            time.sleep(2)
            publish_date = driver.find_element(
                By.CSS_SELECTOR, "div.sticky div.news--author div.post__time"
            ).text
            news_paragraphs = driver.find_elements(
                By.CSS_SELECTOR, "div.news-content-area p"
            )
            news = []
            for news_paragraph in news_paragraphs:
                news.append(news_paragraph.text.replace("\n", " "))
            news = " ".join(news).strip()

            if news == "":
                continue

            new_row = {
                "title": news_titles[index].strip(),
                "news": news.strip(),
                "category": key.strip(),
                "published_date": publish_date.strip(),
            }
            df.loc[len(df)] = new_row

            df.to_csv("dcnepal_news_final2.csv", index=None)

            # with codecs.open('ekantipur_news.csv', 'a', 'utf-8') as file:
            #     file.write(news_titles[index].strip())
            #     file.write("\n")
            #     file.write(news.strip())
            #     file.write("\n")
            #     file.write("प्रवास".strip())
            #     file.write("\n")
            #     file.write(publish_date.strip())
            #     file.write("\n")
            time.sleep(2)
            # driver.back()
            # time.sleep(2)
            # scroll(5)
        except Exception as e:
            continue

    # df.to_csv('ekantipur_news_final2.csv', index=None)
