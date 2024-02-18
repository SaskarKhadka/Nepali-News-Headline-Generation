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

NAGARIKNEWS_WEBSITES = {
    #   "अर्थ / वाणिज्य": "https://nagariknews.nagariknetwork.com/economy",
    # "राजनीति": "https://nagariknews.nagariknetwork.com/politics",
    #   "विचार": "https://nagariknews.nagariknetwork.com/opinion",
    #   "शिक्षा": "https://nagariknews.nagariknetwork.com/education",
    #   "खेलकुद": "https://nagariknews.nagariknetwork.com/sports",
      "विश्व": "https://nagariknews.nagariknetwork.com/international",
    #   "स्वास्थ्य": "https://nagariknews.nagariknetwork.com/health",
    #   "विज्ञान र प्रविधि": "https://nagariknews.nagariknetwork.com/technology",
    #   "समाज": "https://nagariknews.nagariknetwork.com/social-affairs",
    #   "कला": "https://nagariknews.nagariknetwork.com/arts",
    #   "प्रवास": "https://nagariknews.nagariknetwork.com/diaspora"
}

with codecs.open("nagarik_news_links.csv", "r", "utf-8") as file:
    news_links = file.readlines()

with codecs.open("nagarik_news_titles.csv", "r", "utf-8") as file:
    news_titles = file.readlines()


# Go inside each news section, select the news article and save it along with its label
for key, value in NAGARIKNEWS_WEBSITES.items():
    for index, news_cover_link in enumerate(news_links):
        # news_title = news_cover.find_element(By.CSS_SELECTOR, 'div.text h1 a')
        # title = news_title.text
        # driver.execute_script("arguments[0].scrollIntoView();", news_title)
        # driver.execute_script("arguments[0].click();", news_title)
        try:
            driver.get(news_cover_link)
            time.sleep(2)

            # Select paragraph tags except for the last one which determines the published data
            # Some news also have an extra p element with no data. Such data will be analysed and removed after data collection
            news_paragraphs = driver.find_elements(
                By.CSS_SELECTOR, "div.content.text-justify article p"
            )
            publish_date = news_paragraphs[-1].text
            news_paragraphs = news_paragraphs[:-1]
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

            df.to_csv("nagarik_news_final2.csv", index=None)

            # with codecs.open('nagarik_news.csv', 'a', 'utf-8') as file:
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
            continue
