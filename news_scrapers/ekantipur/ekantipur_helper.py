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
options.add_experimental_option('excludeSwitches', ['enable-logging'])

service = Service(chrome_driver)
driver = webdriver.Chrome(service=service, options=options)

news_links = []
news_titles = []

with codecs.open('ekantipur_news_links.csv', 'r', 'utf-8') as file:
    news_links = file.readlines()
    
with codecs.open('ekantipur_news_titles.csv', 'r', 'utf-8') as file:
    news_titles = file.readlines()
    
EKANTIPUR_WEBSITES = {
                #   "अर्थ / वाणिज्य": "https://ekantipur.com/business", 
                #   "विश्व": "https://ekantipur.com/world",
                    # "शिक्षा": "https://ekantipur.com/Education",
                #   "विचार": "https://ekantipur.com/opinion",
                    # "खेलकुद": "https://ekantipur.com/sports",
                #   "स्वास्थ्य": "https://ekantipur.com/health",
                #   "विज्ञान र प्रविधि": "https://ekantipur.com/technology",
                #   "साहित्य/विविध": "https://ekantipur.com/literature",
                #   "अन्य": "https://ekantipur.com/Other",
                  "मनोरञ्जन": "https://ekantipur.com/entertainment",
                #   "कला": "https://ekantipur.com/Art",
                #   "प्रवास": "https://ekantipur.com/diaspora",
                    } 

for key, value in EKANTIPUR_WEBSITES.items():
    for index, news_cover_link in enumerate(news_links):
            # news_title = news_cover.find_element(By.CSS_SELECTOR, 'div.teaser.offset h2 a')
            # title = news_title.text
            # driver.execute_script("arguments[0].scrollIntoView();", news_title)
            # driver.execute_script("arguments[0].click();", news_title)
            try:
                driver.get(news_cover_link)
                time.sleep(2)
                publish_date = driver.find_element(By.CSS_SELECTOR, 'article.normal div.row div.col-xs-12.col-sm-12.col-md-12 time').text
                news_paragraphs = driver.find_elements(By.CSS_SELECTOR, 'div.description.current-news-block p')
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
                
                df.to_csv('ekantipur_news_final2.csv', index=None)
            
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

        