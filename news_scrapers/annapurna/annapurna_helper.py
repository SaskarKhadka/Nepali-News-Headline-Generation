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

ANNAPURNAPOST_WEBSITES = {
    #   "अर्थ / वाणिज्य": "https://annapurnapost.com/category/economy/",
    #   "विचार": "https://annapurnapost.com/category/opinion/",
    # "राजनीति": "https://annapurnapost.com/category/politics/",
    #   "शिक्षा": "https://annapurnapost.com/category/education/",
    #   "खेलकुद": "https://annapurnapost.com/category/sports/",
    #   "विश्व": "https://annapurnapost.com/category/foreign/",
    #   "स्वास्थ्य": "https://annapurnapost.com/category/health/",
    #   "विज्ञान र प्रविधि": "https://annapurnapost.com/category/tech/",
    #   "मनोरञ्जन": "https://annapurnapost.com/category/entertainment/",
    "समाज": "https://annapurnapost.com/category/social/",
    #   "प्रवास": "https://annapurnapost.com/category/prabas/",
}

with codecs.open("annapurna_news_links.csv", "r", "utf-8") as file:
    news_links = file.readlines()

with codecs.open("annapurna_news_titles.csv", "r", "utf-8") as file:
    news_titles = file.readlines()

# Go inside each news section, select the news article and save it along with its label
for key, value in ANNAPURNAPOST_WEBSITES.items():
    for index, news_cover_link in enumerate(news_links):
        # news_title = news_cover.find_element(By.CSS_SELECTOR, 'h3.card__title a')
        # title = news_title.text
        # driver.execute_script("arguments[0].scrollIntoView();", news_title)
        # driver.execute_script("arguments[0].click();", news_title)
        try:
            driver.get(news_cover_link)
            time.sleep(2)

            publish_date = driver.find_element(By.CSS_SELECTOR, "p.date span").text
            news_paragraphs = driver.find_elements(
                By.CSS_SELECTOR, "div.news__details p"
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

            df.to_csv("annapurna_news_final2.csv", index=None)

            # with codecs.open('annapurnapost_news.csv', 'a', 'utf-8') as file:
            #     file.write(news_titles[index].strip())
            #     file.write("\n")
            #     file.write(news.strip())
            #     file.write("\n")
            #     file.write(key.strip())
            #     file.write("\n")
            #     file.write(publish_date.strip())
            #     file.write("\n")
            time.sleep(2)
            # driver.back()
            # time.sleep(2)
        except Exception as e:
            pass
