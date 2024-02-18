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

GORKHAPATRA_WEBSITES = {
    #   "अर्थ / वाणिज्य": "https://gorkhapatraonline.com/categories/economy",
    "राजनीति": "https://gorkhapatraonline.com/categories/politics",
    #   "विचार": "https://gorkhapatraonline.com/categories/thoughts",
    #   "खेलकुद": "https://gorkhapatraonline.com/categories/sports",
    #   "विश्व": "https://gorkhapatraonline.com/categories/international",
    #   "स्वास्थ्य": "https://gorkhapatraonline.com/categories/health",
    # "देश/प्रदेश": "https://gorkhapatraonline.com/categories/province",
    #   "विज्ञान र प्रविधि": "https://gorkhapatraonline.com/categories/technology",
    # "शिक्षा": "https://gorkhapatraonline.com/categories/education",
    #   "मनोरञ्जन": "https://gorkhapatraonline.com/categories/entertainment",
    #   "प्रवास": "https://gorkhapatraonline.com/categories/diaspora",
}

with codecs.open("gorkhapatra_news_links.csv", "r", "utf-8") as file:
    news_links = file.readlines()

with codecs.open("gorkhapatra_news_titles.csv", "r", "utf-8") as file:
    news_titles = file.readlines()

# Go inside each news section, select the news article and save it along with its label
for key, value in GORKHAPATRA_WEBSITES.items():
    for index, news_cover_link in enumerate(news_links):
        # news_title = news_cover.find_element(By.CSS_SELECTOR, 'h2.item-title a')
        # title = news_title.text
        # driver.execute_script("arguments[0].scrollIntoView();", news_title)
        # driver.execute_script("arguments[0].click();", news_title)
        try:
            driver.get(news_cover_link)
            time.sleep(2)

            publish_date = driver.find_element(
                By.CSS_SELECTOR,
                "div.d-flex.justify-content-between.flex-column.flex-md-row div.d-flex.align-items-center.share-inline-block.mb-3 span.mr-3.font-size-16",
            ).text
            news_details = driver.find_elements(
                By.CSS_SELECTOR, "div.single-blog-content div.blog-details"
            )
            news_paragraphs = news_details[-3].find_elements(By.CSS_SELECTOR, "p")
            news = []
            paragraph = 1
            for news_paragraph in news_paragraphs:
                # if paragraph == 1:
                #     paragraph_sentences = news_paragraph.text.replace("\n", " ").split("।")
                #     paragraph_sentences.pop(0)
                #     news = news.append(" । ".join(paragraph_sentences))
                #     paragraph += 1
                news.append(news_paragraph.text.replace("\n", " "))
            news = " ".join(news)
            news = news.split("।")
            news.pop(0)
            news = " । ".join(news)

            if news == "":
                continue

            new_row = {
                "title": news_titles[index].strip(),
                "news": news.strip(),
                "category": key.strip(),
                "published_date": publish_date.strip(),
            }
            df.loc[len(df)] = new_row

            df.to_csv("gorkhapatra_news_final2.csv", index=None)

            # news = news.split("—")
            # with codecs.open('gorkhapatra_news.csv', 'a', 'utf-8') as file:
            #     file.write(news_titles[index].strip())
            #     file.write("\n")
            #     file.write(news.strip())
            #     file.write("\n")
            #     file.write("विचार".strip())
            #     file.write("\n")
            #     file.write(publish_date.strip())
            #     file.write("\n")
            time.sleep(2)
            # driver.back()
            # time.sleep(2)
        except Exception as e:
            continue
