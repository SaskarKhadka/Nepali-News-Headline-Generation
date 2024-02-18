# A web scraper for nepali news sites - Nagarik

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
import codecs
import random

# Path for chrome driver
# check if chrome driver has expired
chrome_driver = "D:/Development/chromedriver.exe"

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

options = webdriver.ChromeOptions()
options.add_experimental_option("excludeSwitches", ["enable-logging"])

service = Service(chrome_driver)
driver = webdriver.Chrome(service=service, options=options)

TOTAL_NEWS_TO_SELECT = 5000


def get_more_news():
    # Get the elements from more-category-news div which are date and articles
    for i in range(250):
        try:
            get_more_category_news = driver.find_element(By.ID, "loadmore")
            driver.execute_script("arguments[0].click();", get_more_category_news)
            time.sleep(1)
        except Exception as e:
            break


for key, value in NAGARIKNEWS_WEBSITES.items():
    driver.get(value)
    time.sleep(2)

    get_more_news()

    # From the list of news select the 150 news articles
    news_list = driver.find_elements(
        By.CSS_SELECTOR, "div.articles article.list-group-item"
    )

    print(len(news_list))

    news_cover_links = []
    news_titles = []

    # break_index = 1000

    # for index in range(len(news_list)):
    #     if "२०८० " in news_list[index].find_element(By.CSS_SELECTOR, 'div.share-wrap div time.npdate').text:
    #         continue
    #     news_list_filt += [news_list[index]]

    # news_list = news_list[:]
    # news_list = news_list[:TOTAL_NEWS_TO_SELECT]

    for news_cover in news_list:
        news_link = news_cover.find_element(By.CSS_SELECTOR, "div.text h1 a")
        news_titles.append(news_link.text)
        news_cover_links.append(news_link.get_attribute("href"))

    with codecs.open("nagarik_news_links.csv", "a", "utf-8") as file:
        file.write("\n".join(news_cover_links))

    with codecs.open("nagarik_news_titles.csv", "a", "utf-8") as file:
        file.write("\n".join(news_titles))

    exit()

    # Go inside each news section, select the news article and save it along with its label
    for index, news_cover_link in enumerate(news_cover_links):
        # news_title = news_cover.find_element(By.CSS_SELECTOR, 'div.text h1 a')
        # title = news_title.text
        # driver.execute_script("arguments[0].scrollIntoView();", news_title)
        # driver.execute_script("arguments[0].click();", news_title)

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
        news = " ".join(news)
        with codecs.open("nagarik_news.csv", "a", "utf-8") as file:
            file.write(news_titles[index].strip())
            file.write("\n")
            file.write(news.strip())
            file.write("\n")
            file.write(key.strip())
            file.write("\n")
            file.write(publish_date.strip())
            file.write("\n")
        time.sleep(2)
        # driver.back()
        # time.sleep(2)
