# A web scraper for nepali news sites - Image Khabar

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
import codecs

# Path for chrome driver
# check if chrome driver has expired
chrome_driver = "D:/Development/chromedriver.exe"

IMAGEKHABAR_WEBSITES = {
    #   "अर्थ / वाणिज्य": "https://www.imagekhabar.com/news/category/economy/",
    #   "विचार": "",
    "राजनीति": "https://www.imagekhabar.com/news/category/politics/",
    #   "शिक्षा": "https://www.imagekhabar.com/news/category/शिक्षा",
    #   "खेलकुद": "https://www.imagekhabar.com/news/category/sports/",
    #   "विश्व": "https://www.imagekhabar.com/news/category/world/",
    #   "स्वास्थ्य": "https://www.imagekhabar.com/news/category/health/",
    #   "विज्ञान र प्रविधि": "https://www.imagekhabar.com/news/category/technology/",
    #   "मनोरञ्जन": "https://www.imagekhabar.com/news/category/entertainment/",
    # "समाज": "https://www.imagekhabar.com/news/category/society/",
    #   "देश/प्रदेश": "https://www.imagekhabar.com/news/category/region/",
}

options = webdriver.ChromeOptions()
options.add_experimental_option("excludeSwitches", ["enable-logging"])

service = Service(chrome_driver)
driver = webdriver.Chrome(service=service, options=options)

# TOTAL_NEWS_TO_SELECT = 210

for key, value in IMAGEKHABAR_WEBSITES.items():
    driver.get(value)
    time.sleep(2)

    total_news = 0

    # Get the news list from the each page until 150 items are selected
    for page in range(1, 150):
        news_list = []
        new_page = value
        if page > 1:
            new_page = value + f"/page/{page}/"
            driver.get(new_page)
            time.sleep(2)

        # banner_news = driver.find_element(By.CSS_SELECTOR, 'div.ok-news-post.ok-samachar-spot-news.flx ')
        # news_list.append(banner_news)
        # category_news = driver.find_elements(By.CSS_SELECTOR, 'div.uk-grid.uk-grid-match.uk-grid-small div div.uk-padding-small.uk-card.uk-card-small.uk-card-default')
        # category_news = driver.find_elements(By.CSS_SELECTOR, 'div.uk-grid.uk-grid-match.uk-grid-small.uk-grid-divider div div.uk-card.latest-item')
        category_news = driver.find_elements(
            By.CSS_SELECTOR,
            "div.uk-first-column div.uk-card.uk-card-default.uk-grid-collapse.uk-margin.uk-grid div div.uk-card-body",
        )
        news_list += category_news

        # total_news += len(news_list)

        # if total_news > TOTAL_NEWS_TO_SELECT:
        #     break

        news_cover_links = []
        news_titles = []
        for news_cover in news_list:
            # news_link = news_cover.find_element(By.CSS_SELECTOR, 'div.uk-card-body h3 a')
            news_link = news_cover.find_element(By.CSS_SELECTOR, "h3.uk-card-title a")
            # news_link = news_cover.find_element(By.CSS_SELECTOR, 'div.uk-margin-small-top h4 a')
            # news_title = news_cover.find_element(By.CSS_SELECTOR, 'div.ok-news-post a div.ok-post-content-wrap h2.ok-news-title-txt')
            news_titles.append(news_link.text)
            news_cover_links.append(news_link.get_attribute("href"))

        with codecs.open("imagekhabar_news_links.csv", "a", "utf-8") as file:
            file.write("\n".join(news_cover_links))
            file.write("\n")

        with codecs.open("imagekhabar_news_titles.csv", "a", "utf-8") as file:
            file.write("\n".join(news_titles))
            file.write("\n")
        continue
        # exit()

        # # Go inside each news section, select the news article and save it along with its label
        # for index, news_cover_link in enumerate(news_cover_links):
        #     # news_title = news_cover.find_element(By.CSS_SELECTOR, 'h3.card__title a')
        #     # title = news_title.text
        #     # driver.execute_script("arguments[0].scrollIntoView();", news_title)
        #     # driver.execute_script("arguments[0].click();", news_title)

        #     driver.get(news_cover_link)
        #     time.sleep(2)

        #     try:
        #         publish_date = driver.find_element(By.CSS_SELECTOR, 'p.date span').text
        #         news_paragraphs = driver.find_elements(By.CSS_SELECTOR, 'div.news__details p')
        #         news = []
        #         for news_paragraph in news_paragraphs:
        #             news.append(news_paragraph.text.replace("\n", " "))
        #         news = " ".join(news)
        #         # news = news.split("—")
        #         with codecs.open('onlinekhabar_news.csv', 'a', 'utf-8') as file:
        #             file.write(news_titles[index].strip())
        #             file.write("\n")
        #             file.write(news.strip())
        #             file.write("\n")
        #             file.write(key.strip())
        #             file.write("\n")
        #             file.write(publish_date.strip())
        #             file.write("\n")
        #         time.sleep(1)
        #         # driver.back()
        #         # time.sleep(2)
        #     except Exception as e:
        #         pass
    # exit()

    # A web scraper for nepali news sites - Image Khabar

# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.common.by import By
# import time
# import codecs

# # Path for chrome driver
# # check if chrome driver has expired
# chrome_driver = "D:/Development/chromedriver.exe"

# IMAGEKHABAR_WEBSITES = {
#                     #   "अर्थ / वाणिज्य": "https://www.imagekhabar.com/news/category/economy/",
#                     #   "विचार": "",
#                     #   "राजनीति": "",
#                     #   "शिक्षा": "",
#                     #   "खेलकुद": "",
#                     #   "विश्व": "https://www.imagekhabar.com/news/category/world/",
#                     #   "स्वास्थ्य": "https://www.imagekhabar.com/news/category/health/",
#                       "विज्ञान र प्रविधि": "https://www.imagekhabar.com/news/category/technology/",
#                     #   "मनोरञ्जन": "",
#                     #   "प्रवास": "",
#                       }

# options = webdriver.ChromeOptions()
# options.add_experimental_option('excludeSwitches', ['enable-logging'])

# service = Service(chrome_driver)
# driver = webdriver.Chrome(service=service, options=options)

# # TOTAL_NEWS_TO_SELECT = 210

# for key, value in IMAGEKHABAR_WEBSITES.items():
#     driver.get(value)
#     time.sleep(2)

#     total_news = 0

#     # Get the news list from the each page until 150 items are selected
#     for page in range(1, 89):
#         news_list = []
#         new_page = value
#         if page > 1:
#             new_page = value + f"/page/{page}/"
#             driver.get(new_page)
#             time.sleep(2)

#         # banner_news = driver.find_element(By.CSS_SELECTOR, 'div.ok-news-post.ok-samachar-spot-news.flx ')
#         # news_list.append(banner_news)
#         # category_news = driver.find_elements(By.CSS_SELECTOR, 'div.uk-grid.uk-grid-match.uk-grid-small div div.uk-padding-small.uk-card.uk-card-small.uk-card-default')
#         category_news = driver.find_elements(By.CSS_SELECTOR, 'div.uk-grid-small.uk-grid-match.uk-grid div div.uk-card')
#         news_list += category_news

#         # total_news += len(news_list)

#         # if total_news > TOTAL_NEWS_TO_SELECT:
#         #     break

#         news_cover_links = []
#         news_titles = []
#         for news_cover in news_list:
#             news_link = news_cover.find_element(By.CSS_SELECTOR, 'div.uk-card-body h3 a.ah')
#             # news_title = news_cover.find_element(By.CSS_SELECTOR, 'div.ok-news-post a div.ok-post-content-wrap h2.ok-news-title-txt')
#             news_titles.append(news_link.text)
#             news_cover_links.append(news_link.get_attribute('href'))

#         with codecs.open('imagekhabar_news_links.csv', 'a', 'utf-8') as file:
#             file.write("\n".join(news_cover_links))
#             file.write("\n")

#         with codecs.open('imagekhabar_news_titles.csv', 'a', 'utf-8') as file:
#             file.write("\n".join(news_titles))
#             file.write("\n")
#         continue
#         # exit()

#         # # Go inside each news section, select the news article and save it along with its label
#         # for index, news_cover_link in enumerate(news_cover_links):
#         #     # news_title = news_cover.find_element(By.CSS_SELECTOR, 'h3.card__title a')
#         #     # title = news_title.text
#         #     # driver.execute_script("arguments[0].scrollIntoView();", news_title)
#         #     # driver.execute_script("arguments[0].click();", news_title)

#         #     driver.get(news_cover_link)
#         #     time.sleep(2)

#         #     try:
#         #         publish_date = driver.find_element(By.CSS_SELECTOR, 'p.date span').text
#         #         news_paragraphs = driver.find_elements(By.CSS_SELECTOR, 'div.news__details p')
#         #         news = []
#         #         for news_paragraph in news_paragraphs:
#         #             news.append(news_paragraph.text.replace("\n", " "))
#         #         news = " ".join(news)
#         #         # news = news.split("—")
#         #         with codecs.open('onlinekhabar_news.csv', 'a', 'utf-8') as file:
#         #             file.write(news_titles[index].strip())
#         #             file.write("\n")
#         #             file.write(news.strip())
#         #             file.write("\n")
#         #             file.write(key.strip())
#         #             file.write("\n")
#         #             file.write(publish_date.strip())
#         #             file.write("\n")
#         #         time.sleep(1)
#         #         # driver.back()
#         #         # time.sleep(2)
#         #     except Exception as e:
#         #         pass
#     # exit()
