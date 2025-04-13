from dataclasses import dataclass
from time import sleep
import time
from selenium.webdriver import Chrome
import random
from . import constants as c

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC


@dataclass
class Contact:
    name: str = None
    occupation: str = None
    url: str = None


@dataclass
class Institution:
    institution_name: str = None
    linkedin_url: str = None
    website: str = None
    industry: str = None
    type: str = None
    headquarters: str = None
    company_size: int = None
    founded: int = None


@dataclass
class Experience(Institution):
    from_date: str = None
    to_date: str = None
    description: str = None
    position_title: str = None
    duration: str = None
    location: str = None


@dataclass
class Education(Institution):
    from_date: str = None
    to_date: str = None
    description: str = None
    degree: str = None


@dataclass
class Interest(Institution):
    title = None


@dataclass
class Accomplishment(Institution):
    category = None
    title = None


@dataclass
class Scraper:
    driver: Chrome = None
    WAIT_FOR_ELEMENT_TIMEOUT = 5
    TOP_CARD = "pv-top-card"

    @staticmethod
    def wait(duration):
        sleep(int(duration))

    def focus(self):
        self.driver.execute_script('alert("Focus window")')
        self.driver.switch_to.alert.accept()

    def mouse_click(self, elem):
        action = webdriver.ActionChains(self.driver)
        action.move_to_element(elem).perform()

    def wait_for_element_to_load(self, by=By.CLASS_NAME, name="pv-top-card", base=None):
        base = base or self.driver
        return WebDriverWait(base, self.WAIT_FOR_ELEMENT_TIMEOUT).until(
            EC.presence_of_element_located(
                (
                    by,
                    name
                )
            )
        )

    def wait_for_all_elements_to_load(self, by=By.CLASS_NAME, name="pv-top-card", base=None):
        base = base or self.driver
        return WebDriverWait(base, self.WAIT_FOR_ELEMENT_TIMEOUT).until(
            EC.presence_of_all_elements_located(
                (
                    by,
                    name
                )
            )
        )


    def is_signed_in(self):
        try:
            WebDriverWait(self.driver, self.WAIT_FOR_ELEMENT_TIMEOUT).until(
                EC.presence_of_element_located(
                    (
                        By.CLASS_NAME,
                        c.VERIFY_LOGIN_ID,
                    )
                )
            )

            self.driver.find_element(By.CLASS_NAME, c.VERIFY_LOGIN_ID)
            return True
        except Exception as e:
            pass
        return False

    def scroll_to_half(self):
        self.driver.execute_script(
            "window.scrollTo(0, Math.ceil(document.body.scrollHeight/2));"
        )

    def scroll_to_bottom(self):
        self.driver.execute_script(
            "window.scrollTo(0, document.body.scrollHeight);"
        )

    def scroll_to_bottom(self, pause_time=2, max_attempts=10, min_job_count=25, scroll_button_value=None):
        try:
            scrollable_element = self.driver.find_element(By.CLASS_NAME, scroll_button_value)
        except Exception as e:
            print(f"[ERROR] Could not find scrollable container: {e}")
            return

        scroll_height = self.driver.execute_script("return arguments[0].scrollHeight", scrollable_element)
        scroll_position = 0
        attempts = 0

        while attempts < max_attempts:
            # Get current number of job listings
            job_cards = self.driver.find_elements(By.CSS_SELECTOR, "li.jobs-search-results__list-item")
            current_job_count = len(job_cards)

            if current_job_count >= min_job_count:
                print(f"[INFO] Found {current_job_count} job listings. Continuing to scrape...")
            else:
                print(f"[INFO] Found only {current_job_count} listings. Scrolling more...")

            # Scroll by 10% of total height
            scroll_position += int(scroll_height * 0.10)
            self.driver.execute_script("arguments[0].scrollTop = arguments[1]", scrollable_element, scroll_position)
            print(f"[DEBUG] Scrolled to {scroll_position}px")

            # Pause for new content to load
            time.sleep(random.uniform(pause_time, pause_time + 2))

            # Update scroll height in case it dynamically grows
            new_scroll_height = self.driver.execute_script("return arguments[0].scrollHeight", scrollable_element)

            if scroll_position >= new_scroll_height:
                print("[INFO] Reached the bottom or no more content to scroll.")
                break

            scroll_height = new_scroll_height
            attempts += 1


    def scroll_class_name_element_to_page_percent(self, class_name:str, page_percent:float):
        self.driver.execute_script(
            f'elem = document.getElementsByClassName("{class_name}")[0]; elem.scrollTo(0, elem.scrollHeight*{str(page_percent)});'
        )

    def __find_element_by_class_name__(self, class_name):
        try:
            self.driver.find_element(By.CLASS_NAME, class_name)
            return True
        except:
            pass
        return False

    def __find_element_by_xpath__(self, tag_name):
        try:
            self.driver.find_element(By.XPATH,tag_name)
            return True
        except:
            pass
        return False

    def __find_enabled_element_by_xpath__(self, tag_name):
        try:
            elem = self.driver.find_element(By.XPATH,tag_name)
            return elem.is_enabled()
        except:
            pass
        return False

    @classmethod
    def __find_first_available_element__(cls, *args):
        for elem in args:
            if elem:
                return elem[0]
