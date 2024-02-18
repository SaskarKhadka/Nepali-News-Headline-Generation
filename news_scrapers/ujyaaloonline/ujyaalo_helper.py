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

UJYAALO_WEBSITES = {
                        # "अर्थ / वाणिज्य": "https://ujyaaloonline.com/category/economy", 
                    #   "राजनीति": "https://ujyaaloonline.com/category/politics",                    
                    #   "विचार": "https://gorkhapatraonline.com/categories/thoughts",
                    #   "खेलकुद": "https://gorkhapatraonline.com/categories/sports",
                    #   "विश्व": "https://gorkhapatraonline.com/categories/international",
                    #   "स्वास्थ्य": "https://gorkhapatraonline.com/categories/health",
                    #   "विज्ञान र प्रविधि": "https://gorkhapatraonline.com/categories/technology",
                        "मनोरञ्जन": "https://ujyaaloonline.com/category/entertainment",
                        # "समाज": "https://ujyaaloonline.com/category/social",
                    #   "प्रवास": "https://gorkhapatraonline.com/categories/diaspora",
                      } 

with codecs.open('ujyaalo_news_links.csv', 'r', 'utf-8') as file:
    news_links = file.readlines()
    
with codecs.open('ujyaalo_news_titles.csv', 'r', 'utf-8') as file:
    news_titles = file.readlines()
    
# Go inside each news section, select the news article and save it along with its label
for key, value in UJYAALO_WEBSITES.items():
    for index, news_cover_link in enumerate(news_links):
        
        # news_title = news_cover.find_element(By.CSS_SELECTOR, 'h2.item-title a')
        # title = news_title.text
        # driver.execute_script("arguments[0].scrollIntoView();", news_title)
        # driver.execute_script("arguments[0].click();", news_title)
        try:
            driver.get(news_cover_link)
            time.sleep(2)
            
            publish_date = driver.find_element(By.XPATH, '/html/body/section/div/div/div[3]/div[1]/div[1]/div[1]').text
            news_paragraphs = driver.find_elements(By.CSS_SELECTOR, 'div#body-content p')
            news = []
            paragraph = 1
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
            
            df.to_csv('ujyaalo_news_final2.csv', index=None)
            
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
            print("hi")
            continue