import os
import time
from typing import List
from time import sleep
import urllib.parse

from .objects import Scraper
from . import constants as c
from .jobs import Job

from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys


class JobSearch(Scraper):
    AREAS = ["recommended_jobs", None, "still_hiring", "more_jobs"]

    def __init__(self, driver, base_url="https://www.linkedin.com/jobs/", close_on_complete=False, scrape=True, scrape_recommended_jobs=True):
        super().__init__()
        self.driver = driver
        self.base_url = base_url

        if scrape:
            self.scrape(close_on_complete, scrape_recommended_jobs)


    def scrape(self, close_on_complete=True, scrape_recommended_jobs=True):
        if self.is_signed_in():
            self.scrape_logged_in(close_on_complete=close_on_complete, scrape_recommended_jobs=scrape_recommended_jobs)
        else:
            raise NotImplemented("This part is not implemented yet")


    # def scrape_job_card(self, base_element) -> Job:
    #     job_div = self.wait_for_element_to_load(name="job-card-list__title", base=base_element)
    #     job_title = job_div.text.strip()
    #     linkedin_url = job_div.get_attribute("href")
    #     company = base_element.find_element_by_class_name("artdeco-entity-lockup__subtitle").text
    #     location = base_element.find_element_by_class_name("job-card-container__metadata-wrapper").text
    #     job = Job(linkedin_url=linkedin_url, job_title=job_title, company=company, location=location, scrape=False, driver=self.driver)
    #     return job

    # def scrape_job_card(self, base_element) -> Job:
    #     try:
    #         # Extract job title - updated selector
    #         title_elem = base_element.find_element(By.CSS_SELECTOR, "a.job-card-list__title-link")
    #         job_title = title_elem.text.strip()
            
    #         # Extract company - updated selector
    #         company_elem = base_element.find_element(By.CSS_SELECTOR, ".artdeco-entity-lockup__subtitle span")
    #         company = company_elem.text.strip()
            
    #         # Extract location - updated selector
    #         location_elem = base_element.find_element(By.CSS_SELECTOR, ".job-card-container__metadata-item")
    #         location = location_elem.text.strip()
            
    #         # Extract URL
    #         linkedin_url = title_elem.get_attribute("href")
            
    #         # Extract salary if available - new selector
    #         salary = None
    #         try:
    #             salary_elem = base_element.find_element(By.CSS_SELECTOR, ".job-card-container__metadata-item:nth-child(2)")
    #             salary = salary_elem.text.strip()
    #         except:
    #             pass
                
    #         # Create Job object without salary if not needed
    #         job = Job(
    #             linkedin_url=linkedin_url,
    #             job_title=job_title,
    #             company=company,
    #             location=location,
    #             salary=salary,  # Now this will work
    #             scrape=False,
    #             driver=self.driver
    #         )
    #         return job
            
    #     except Exception as e:
    #         print(f"Error scraping job card: {e}")
    #         return None

    def scrape_job_card(self, base_element):
        try:
            # Job Title - multiple selector options with different approaches
            job_title = None
            linkedin_url = None
            
            # Try multiple approaches to get job title
            for approach in [
                # Approach 1: Get from aria-label attribute
                lambda: base_element.find_element(By.CSS_SELECTOR, "[aria-label]").get_attribute("aria-label"),
                
                # Approach 2: Get from visible text in title link
                lambda: base_element.find_element(
                    By.CSS_SELECTOR, "a.job-card-container__link, a.job-card-list__title-link"
                ).text.strip(),
                
                # Approach 3: Get from hidden span (some titles are in visually-hidden spans)
                lambda: base_element.find_element(
                    By.CSS_SELECTOR, "span.visually-hidden"
                ).text.strip(),
                
                # Approach 4: Get from any h3 element (common title container)
                lambda: base_element.find_element(By.CSS_SELECTOR, "h3").text.strip()
            ]:
                try:
                    job_title = approach()
                    if job_title and len(job_title) > 3:  # Basic validation
                        break
                except:
                    continue

            # Get URL - try multiple selectors
            for selector in [
                "a.job-card-container__link",
                "a.job-card-list__title-link",
                "a[data-control-id]"
            ]:
                try:
                    link = base_element.find_element(By.CSS_SELECTOR, selector)
                    linkedin_url = link.get_attribute("href")
                    break
                except:
                    continue

            if not job_title:
                print("Could not extract job title from element")
                return None

            # Company - multiple selector options
            company = ""
            for selector in [
                ".artdeco-entity-lockup__subtitle span",
                ".job-card-container__company-name",
                ".job-card-container__primary-description"
            ]:
                try:
                    company = base_element.find_element(By.CSS_SELECTOR, selector).text.strip()
                    break
                except:
                    continue

            # Location - multiple selector options
            location = ""
            for selector in [
                ".job-card-container__metadata-wrapper li:first-child",
                ".job-card-container__metadata-item",
                ".job-card-container__location"
            ]:
                try:
                    location = base_element.find_element(By.CSS_SELECTOR, selector).text.strip()
                    break
                except:
                    continue

            # Salary - optional field
            salary = None
            for selector in [
                ".job-card-container__metadata-wrapper li:nth-child(2)",
                ".job-card-container__salary-info",
                ".job-card-container__metadata-item--salary"
            ]:
                try:
                    salary = base_element.find_element(By.CSS_SELECTOR, selector).text.strip()
                    break
                except:
                    continue

            # Create Job object
            job = Job(
                linkedin_url=linkedin_url,
                job_title=job_title,
                company=company,
                location=location,
                salary=salary,
                scrape=False,
                driver=self.driver
            )
            return job
            
        except Exception as e:
            print(f"Error scraping job card: {e}")
            return None
    


    def scrape_logged_in(self, close_on_complete=True, scrape_recommended_jobs=True):
        driver = self.driver
        driver.get(self.base_url)
        if scrape_recommended_jobs:
            self.focus()
            sleep(self.WAIT_FOR_ELEMENT_TIMEOUT)
            job_area = self.wait_for_element_to_load(name="scaffold-finite-scroll__content")
            areas = self.wait_for_all_elements_to_load(name="artdeco-card", base=job_area)
            for i, area in enumerate(areas):
                area_name = self.AREAS[i]
                if not area_name:
                    continue
                area_results = []
                for job_posting in area.find_elements_by_class_name("jobs-job-board-list__item"):
                    job = self.scrape_job_card(job_posting)
                    area_results.append(job)
                setattr(self, area_name, area_results)
        return


    # def search(self, search_term: str) -> List[Job]:
    #     url = os.path.join(self.base_url, "search") + f"?keywords={urllib.parse.quote(search_term)}&refresh=true"
    #     self.driver.get(url)
    #     self.scroll_to_bottom()
    #     self.focus()
    #     sleep(self.WAIT_FOR_ELEMENT_TIMEOUT)

    #     job_listing_class_name = "jobs-search-results-list"
    #     job_listing = self.wait_for_element_to_load(name=job_listing_class_name)

    #     self.scroll_class_name_element_to_page_percent(job_listing_class_name, 0.3)
    #     self.focus()
    #     sleep(self.WAIT_FOR_ELEMENT_TIMEOUT)

    #     self.scroll_class_name_element_to_page_percent(job_listing_class_name, 0.6)
    #     self.focus()
    #     sleep(self.WAIT_FOR_ELEMENT_TIMEOUT)

    #     self.scroll_class_name_element_to_page_percent(job_listing_class_name, 1)
    #     self.focus()
    #     sleep(self.WAIT_FOR_ELEMENT_TIMEOUT)

    #     job_results = []
    #     for job_card in self.wait_for_all_elements_to_load(name="job-card-list", base=job_listing):
    #         job = self.scrape_job_card(job_card)
    #         job_results.append(job)
    #     return job_results
    def search(self, search_term: str) -> List[Job]:
        url = f"{self.base_url}search/?keywords={urllib.parse.quote(search_term)}"
        self.driver.get(url)
        
        # Wait for page to load completely
        time.sleep(5)
        
        # Scroll to trigger lazy loading
        self.scroll_to_bottom()
        time.sleep(2)
        
        # Find all job cards using updated selector
        job_cards = []
        try:
            job_cards = self.driver.find_elements(
                By.CSS_SELECTOR, 
                "li.jobs-search-results__list-item"
            )
        except:
            try:
                job_cards = self.driver.find_elements(
                    By.CSS_SELECTOR,
                    "li.scaffold-layout__list-item"
                )
            except Exception as e:
                print(f"Could not find job cards: {e}")
                return []
        
        job_results = []
        for card in job_cards:
            try:
                job = self.scrape_job_card(card)
                if job:
                    job_results.append(job)
            except Exception as e:
                print(f"Failed to process a job card: {e}")
                continue
                
        return job_results
