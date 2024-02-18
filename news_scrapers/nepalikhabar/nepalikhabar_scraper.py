# A web scraper for nepali news sites - Ujyaalo Online

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
import codecs

# Path for chrome driver
# check if chrome driver has expired
chrome_driver = "D:/Development/chromedriver.exe"

NEPALIKHABAR_WEBSITES = {
                    #   "राजनीति": "https://nepalkhabar.com/category/politics",
                    #   "अर्थ / वाणिज्य": "https://nepalkhabar.com/category/economy",
                    #   "विचार": "https://gorkhapatraonline.com/categories/thoughts",
                    #   "खेलकुद": "https://nepalkhabar.com/category/sports",
                    #   "विश्व": "https://gorkhapatraonline.com/categories/international",
                        # "शिक्षा": "https://www.ratopati.com/category/education",
                    #   "स्वास्थ्य": "https://gorkhapatraonline.com/categories/health",
                        "समाज": "https://nepalkhabar.com/category/society",
                    #   "विज्ञान र प्रविधि": "https://gorkhapatraonline.com/categories/technology",
                    #   "मनोरञ्जन": "https://www.ratopati.com/category/entertainment",
                    #   "प्रवास": "https://gorkhapatraonline.com/categories/diaspora",
                    } 

options = webdriver.ChromeOptions()
options.add_experimental_option('excludeSwitches', ['enable-logging'])

service = Service(chrome_driver)
driver = webdriver.Chrome(service=service, options=options)

# TOTAL_NEWS_TO_SELECT = 150

for key, value in NEPALIKHABAR_WEBSITES.items():
    driver.get(value)
    time.sleep(2)
    

    # Get the news list from the each page until 150 items are selected
    for page in range(800, 1140):
        news_list = []
        new_page = value
        try:
            if page > 1:
                new_page = value + f"?page={page}"
                driver.get(new_page)
                time.sleep(2)
            # banner_news = driver.find_element(By.CSS_SELECTOR, 'div.grid-posts div.col-md-4.padding-10 div.grid-post')
            # news_list.append(banner_news)
            category_news = driver.find_elements(By.CSS_SELECTOR, 'ul.uk-list.uk-list-large.uk-list-divider li.nk-item-list-first-page div.uk-grid.uk-grid-medium')
            # category_news = driver.find_elements(By.CSS_SELECTOR, 'ul.uk-list.uk-list-large.uk-list-divider li div.uk-grid.uk-flex.uk-flex-top')
            news_list += category_news
            
            news_cover_links = []
            news_titles = []
            for news_cover in news_list:
                news_link = news_cover.find_element(By.CSS_SELECTOR, 'div.uk-first-column h3.uk-margin-remove a.uk-link-reset')
                # news_title = news_link.find_element(By.CSS_SELECTOR, 'div.items a').text
                news_titles.append(news_link.text)
                news_cover_links.append(news_link.get_attribute('href'))
            
            with codecs.open('nepalikhabar_news_links.csv', 'a', 'utf-8') as file:
                file.write("\n".join(news_cover_links))
                file.write("\n")
                
            with codecs.open('nepalikhabar_news_titles.csv', 'a', 'utf-8') as file:
                    file.write("\n".join(news_titles))
                    file.write("\n")
                    
                    
            continue
        except Exception as e:
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