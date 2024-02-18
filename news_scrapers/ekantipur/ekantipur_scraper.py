# A web scraper for nepali news sites - Ekantipur

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
import codecs

# Path for chrome driver
# check if chrome driver has expired
chrome_driver = "D:/Development/chromedriver.exe"

EKANTIPUR_WEBSITES = {
                    #   "अर्थ / वाणिज्य": "https://ekantipur.com/business", 
                    #   "विश्व": "https://ekantipur.com/world",
                    #   "विचार": "https://ekantipur.com/opinion",
                    #   "खेलकुद": "https://ekantipur.com/sports",
                      "स्वास्थ्य": "https://ekantipur.com/health",
                    #   "शिक्षा": "https://ekantipur.com/Education",
                    #   "विज्ञान र प्रविधि": "https://ekantipur.com/technology",
                    #   "साहित्य/विविध": "https://ekantipur.com/literature",
                    #   "अन्य": "https://ekantipur.com/Other",
                    #   "मनोरञ्जन": "https://ekantipur.com/entertainment",
                    #   "कला": "https://ekantipur.com/Art",
                    #   "प्रवास": "https://ekantipur.com/diaspora",
                      } 

options = webdriver.ChromeOptions()
options.add_experimental_option('excludeSwitches', ['enable-logging'])

service = Service(chrome_driver)
driver = webdriver.Chrome(service=service, options=options)

TOTAL_NEWS_TO_SELECT = 4750

def scroll(total=500):
    # Scroll to the bottom of the page so that a lot of more-catgeory-news div is loaded
    scroll_count = 0
    while(scroll_count < total):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)
        scroll_count += 1

for key, value in EKANTIPUR_WEBSITES.items():
    driver.get(value)
    time.sleep(2)
    scroll()

    # Get the elements from more-category-news div which are date and articles
    all_news = driver.find_element(By.CLASS_NAME, 'more-category-news')
    
    more_category_news = all_news.find_elements(By.CSS_SELECTOR, 'article.normal')

    # for i in range(0, 30):
    #     more_category_news += all_news.find_elements(By.CSS_SELECTOR, 'article.normal')
    #     if len(more_category_news) > TOTAL_NEWS_TO_SELECT:
    #         break
    #     all_news = all_news.find_element(By.CLASS_NAME, 'more-category-news')

    print(len(more_category_news))

    # From the more-category-news select the top 150 news articles  
    # news_list = more_category_news[:TOTAL_NEWS_TO_SELECT]
    news_list = more_category_news
    # news_list = more_category_news
    # news_list = []
    # for category_news in more_category_news:
    #     news = category_news.find_elements(By.CLASS_NAME, 'normal')
    #     if len(news_list) <= TOTAL_NEWS_TO_SELECT and len(news_list) + len(news) <= TOTAL_NEWS_TO_SELECT:
    #         news_list += news
    #     elif len(news_list) + len(news) > TOTAL_NEWS_TO_SELECT:
    #         needed_news_count = TOTAL_NEWS_TO_SELECT - len(news_list)
    #         news_list += news[:needed_news_count]
    #     else:
    #         break
    
    news_cover_links = []
    news_titles = []
    for news_cover in news_list:
        try:
            news_link = news_cover.find_element(By.CSS_SELECTOR, 'div.teaser.offset h2 a')
            news_titles.append(news_link.text)
            news_cover_links.append(news_link.get_attribute('href'))
        except Exception as e:
            continue
        
    with codecs.open('ekantipur_news_links.csv', 'a', 'utf-8') as file:
            file.write("\n".join(news_cover_links))
            
    with codecs.open('ekantipur_news_titles.csv', 'a', 'utf-8') as file:
            file.write("\n".join(news_titles))
            
    exit()

    # Go inside each news section, select the news article and save it along with its label
    for index, news_cover_link in enumerate(news_cover_links):
        # news_title = news_cover.find_element(By.CSS_SELECTOR, 'div.teaser.offset h2 a')
        # title = news_title.text
        # driver.execute_script("arguments[0].scrollIntoView();", news_title)
        # driver.execute_script("arguments[0].click();", news_title)
        driver.get(news_cover_link)
        time.sleep(2)
        publish_date = driver.find_element(By.CSS_SELECTOR, 'article.normal div.row div.col-xs-10.col-sm-10.col-md-10 time').text
        news_paragraphs = driver.find_elements(By.CSS_SELECTOR, 'div.description.current-news-block p')
        news = []
        for news_paragraph in news_paragraphs:
            news.append(news_paragraph.text.replace("\n", " "))
        news = " ".join(news)
        with codecs.open('ekantipur_news.csv', 'a', 'utf-8') as file:
            file.write(news_titles[index].strip())
            file.write("\n")
            file.write(news.strip())
            file.write("\n")
            file.write(key.strip())
            file.write("\n")
            file.write(publish_date.strip())
            file.write("\n")
        time.sleep(5)
        # driver.back()
        # time.sleep(2)
        # scroll(5)