# A web scraper for nepali news sites - Ujyaalo Online

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
import codecs

# Path for chrome driver
# check if chrome driver has expired
chrome_driver = "D:/Development/chromedriver.exe"

RATOPATI_WEBSITES = {
                    #   "अर्थ / वाणिज्य": "https://www.ratopati.com/category/economy", 
                    #   "राजनीति": "https://gorkhapatraonline.com/categories/politics",
                    #   "विचार": "https://gorkhapatraonline.com/categories/thoughts",
                    #   "खेलकुद": "https://www.ratopati.com/category/sports",
                    # "देश/प्रदेश": "https://sudurpashchim.ratopati.com/province/7",
                    #   "विश्व": "https://www.ratopati.com/category/international",
                        # "शिक्षा": "https://www.ratopati.com/category/education",
                    #   "स्वास्थ्य": "https://gorkhapatraonline.com/categories/health",
                    #   "विज्ञान र प्रविधि": "https://gorkhapatraonline.com/categories/technology",
                      "मनोरञ्जन": "https://www.ratopati.com/category/entertainment",
                    #   "प्रवास": "https://gorkhapatraonline.com/categories/diaspora",
                    } 

options = webdriver.ChromeOptions()
options.add_experimental_option('excludeSwitches', ['enable-logging'])

service = Service(chrome_driver)
driver = webdriver.Chrome(service=service, options=options)

# TOTAL_NEWS_TO_SELECT = 150

for key, value in RATOPATI_WEBSITES.items():
    driver.get(value)
    time.sleep(2)
    

    # Get the news list from the each page until 150 items are selected
    for page in range(1, 366):
        news_list = []
        new_page = value
        if page > 1:
            new_page = value + f"?page={page}"
            driver.get(new_page)
            time.sleep(2)
            
        # banner_news = driver.find_element(By.CSS_SELECTOR, 'div.grid-posts div.col-md-4.padding-10 div.grid-post')
        # news_list.append(banner_news)
        category_news = driver.find_elements(By.CSS_SELECTOR, 'div.dn-grid div.columnnews.mbl-col.col3')
        news_list += category_news
        
        news_cover_links = []
        news_titles = []
        for news_cover in news_list:
            news_link = news_cover.find_element(By.CSS_SELECTOR, 'a')
            news_title = news_link.find_element(By.CSS_SELECTOR, 'div.columnnews-wrap h3').text
            news_titles.append(news_title)
            news_cover_links.append(news_link.get_attribute('href'))
        
        with codecs.open('ratopati_news_links.csv', 'a', 'utf-8') as file:
            file.write("\n".join(news_cover_links))
            file.write("\n")
            
        with codecs.open('ratopati_news_titles.csv', 'a', 'utf-8') as file:
                file.write("\n".join(news_titles))
                file.write("\n")
                
                
        continue
        
        # Go inside each news section, select the news article and save it along with its label
        for index, news_cover_link in enumerate(news_cover_links):
            
            # news_title = news_cover.find_element(By.CSS_SELECTOR, 'h2.item-title a')
            # title = news_title.text
            # driver.execute_script("arguments[0].scrollIntoView();", news_title)
            # driver.execute_script("arguments[0].click();", news_title)
            
            driver.get(news_cover_link)
            time.sleep(2)
            
            publish_date = driver.find_element(By.CSS_SELECTOR, 'div.d-flex.justify-content-between.flex-column.flex-md-row div.d-flex.align-items-center.share-inline-block.mb-3 span.mr-3.font-size-16').text
            news_details = driver.find_elements(By.CSS_SELECTOR, 'div.single-blog-content div.blog-details')
            news_paragraphs = news_details[-3].find_elements(By.CSS_SELECTOR, 'p')
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
            
            # news = news.split("—")
            with codecs.open('gorkhapatra_news.csv', 'a', 'utf-8') as file:
                file.write(news_titles[index].strip())
                file.write("\n")
                file.write(news.strip())
                file.write("\n")
                file.write(key.strip())
                file.write("\n")
                file.write(publish_date.strip())
                file.write("\n")
            time.sleep(1)
            # driver.back()
            # time.sleep(2)