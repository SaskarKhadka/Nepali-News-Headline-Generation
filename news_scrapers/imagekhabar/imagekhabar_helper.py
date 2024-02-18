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

IMAGEKHABAR_WEBSITES = {
    #   "अर्थ / वाणिज्य": "https://www.imagekhabar.com/news/category/economy/",
    #   "विचार": "",
    # "समाज": "https://www.imagekhabar.com/news/category/society/",
    "राजनीति": "https://www.imagekhabar.com/news/category/politics/",
    #   "शिक्षा": "https://www.imagekhabar.com/news/category/शिक्षा",
    # "खेलकुद": "https://www.imagekhabar.com/news/category/sports/",
    #   "विश्व": "https://www.imagekhabar.com/news/category/world/",
    #   "स्वास्थ्य": "https://www.imagekhabar.com/news/category/health/",
    #   "विज्ञान र प्रविधि": "https://www.imagekhabar.com/news/category/technology/",
    #   "मनोरञ्जन": "https://www.imagekhabar.com/news/category/entertainment/",
    # "समाज": "https://www.imagekhabar.com/news/category/society/",
    # "देश/प्रदेश": "https://www.imagekhabar.com/news/category/region/",
}

with codecs.open("imagekhabar_news_links.csv", "r", "utf-8") as file:
    news_links = file.readlines()

with codecs.open("imagekhabar_news_titles.csv", "r", "utf-8") as file:
    news_titles = file.readlines()


# Go inside each news section, select the news article and save it along with its label
for key, value in IMAGEKHABAR_WEBSITES.items():
    for index, news_cover_link in enumerate(news_links):
        # news_title = news_cover.find_element(By.CSS_SELECTOR, 'h3.card__title a')
        # title = news_title.text
        # driver.execute_script("arguments[0].scrollIntoView();", news_title)
        # driver.execute_script("arguments[0].click();", news_title)

        driver.get(news_cover_link)
        time.sleep(2)
        # skip_this = driver.find_element(By.XPATH, '/html/body/div[1]/div/div/div/div[2]/span')
        # driver.execute_script("arguments[0].click();", skip_this)
        # time.sleep(1)

        try:
            publish_date = driver.find_element(
                By.CSS_SELECTOR, "div.post-time p.single-date"
            ).text
            news_paragraphs = driver.find_elements(
                By.CSS_SELECTOR, "article.post-entry p"
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

            df.to_csv("imagekhabar_news_final2.csv", index=None)

            # with codecs.open('imagekhabar_news.csv', 'a', 'utf-8') as file:
            #     file.write(news_titles[index].strip())
            #     file.write("\n")
            #     file.write(news.strip())
            #     file.write("\n")
            #     file.write("शिक्षा".strip())
            #     file.write("\n")
            #     file.write(publish_date.strip())
            #     file.write("\n")
            time.sleep(2)
            # driver.back()
            # time.sleep(2)
        except Exception as e:
            print("hi")
            pass
