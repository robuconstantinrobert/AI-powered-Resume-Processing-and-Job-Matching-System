###############################################################################################################
################################## WORKING SCRAPING SCRIPT FOR USER PROFILES ##################################

# from linkedin_scraper import Person, actions
# from selenium import webdriver
# import time

# # Configure Chrome options
# options = webdriver.ChromeOptions()
# options.add_argument("--disable-gpu")  # Disable GPU hardware acceleration
# options.add_argument("--disable-software-rasterizer")  # Disable software fallback
# options.add_argument("--disable-dev-shm-usage")  # Prevent /dev/shm issues
# options.add_argument("--no-sandbox")  # Bypass OS security (useful in Docker)
# options.add_argument("--disable-extensions")  # Disable extensions
# options.add_argument("--disable-infobars")  # Disable infobars
# options.add_argument("--disable-blink-features=AutomationControlled")  # Hide automation
# options.add_experimental_option('excludeSwitches', ['enable-logging'])  # Suppress logs

# print("Initializing Chrome driver...")
# driver = webdriver.Chrome(options=options)

# try:
#     # LinkedIn credentials (leave empty for terminal prompt)
#     email = "robuconstantinrobert@gmail.com"
#     password = "contgmail123456789"

#     print("Logging in to LinkedIn...")
#     actions.login(driver, email, password)
#     time.sleep(3)  # Wait for login to complete

#     # Profile URL to scrape
#     profile_url = "https://www.linkedin.com/in/ruxandraalexandreanu"
#     print(f"Scraping profile: {profile_url}")

#     # Initialize Person with detailed scraping
#     person = Person(
#         linkedin_url=profile_url,
#         driver=driver,
#         scrape=True,
#         close_on_complete=False  # Keep browser open to inspect
#     )

#     # Print basic info
#     print("\n=== BASIC INFO ===")
#     print(f"Name: {person.name}")
#     print(f"About: {person.about}\n")

#     # Print Experience in detail
#     print("\n=== EXPERIENCE ===")
#     if person.experiences:
#         for idx, exp in enumerate(person.experiences, 1):
#             print(f"\nExperience #{idx}:")
#             print(f"  Title: {exp.position_title}")
#             print(f"  Company: {exp.institution_name}")
#             print(f"  Duration: {exp.from_date} - {exp.to_date or 'Present'}")
#             print(f"  Location: {exp.location}")
#             print(f"  Description: {exp.description}")
#             print(f"  LinkedIn URL: {exp.linkedin_url}")
#     else:
#         print("No experience found.")

#     # Print Education in detail
#     print("\n=== EDUCATION ===")
#     if person.educations:
#         for idx, edu in enumerate(person.educations, 1):
#             print(f"\nEducation #{idx}:")
#             print(f"  Degree: {edu.degree}")
#             print(f"  Institution: {edu.institution_name}")
#             print(f"  Duration: {edu.from_date} - {edu.to_date or 'Present'}")
#             print(f"  Description: {edu.description}")
#             print(f"  LinkedIn URL: {edu.linkedin_url}")
#     else:
#         print("No education found.")

#     # Print additional sections if available
#     if person.interests:
#         print("\n=== INTERESTS ===")
#         for interest in person.interests:
#             print(f"- {interest}")

#     if person.accomplishments:
#         print("\n=== ACCOMPLISHMENTS ===")
#         for acc in person.accomplishments:
#             print(f"- {acc}")

# except Exception as e:
#     print(f"\nError during scraping: {str(e)}")
#     # Take screenshot for debugging
#     driver.save_screenshot("error_screenshot.png")
#     print("Saved screenshot to 'error_screenshot.png'")

# finally:
#     input("\nPress Enter to close the browser...")  # Pause to review output
#     driver.quit()
#     print("Browser closed.")

###############################################################################################################
###############################################################################################################



###############################################################################################################
################################## WORKING SCRAPING SCRIPT FOR USER PROFILES ##################################

import os
import urllib
import time
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from linkedin_scraper import JobSearch, actions
from linkedin_scraper.job_search import JobSearch as OriginalJobSearch
from linkedin_scraper.jobs import Job

# Configure Chrome options
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--disable-blink-features=AutomationControlled")
#chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
chrome_options.add_argument("--disable-gpu")

driver = webdriver.Chrome(options=chrome_options)
email = "@gmail.com"
password = ""

class FixedJobSearch(OriginalJobSearch):
    def scrape_job_card(self, base_element):
        try:
            # Job Title - multiple selector options
            title_elem = None
            for selector in [
                "a.job-card-list__title-link",  # Primary selector
                "a.job-card-container__link",   # Fallback 1
                "a.jobs-unified-top-card__job-title-link",  # Fallback 2
                "a.job-card-list__title",
                "a.job-card-list__title--link",
                "job-card-list__title--link"
            ]:
                try:
                    title_elem = base_element.find_element(By.CSS_SELECTOR, selector)
                    break
                except:
                    continue
            
            if not title_elem:
                print("Could not find job title element")
                return None

            job_title = title_elem.get_attribute("aria-label") or title_elem.text.strip()
            linkedin_url = title_elem.get_attribute("href")

            # Company - multiple selector options
            company = ""
            for selector in [
                ".artdeco-entity-lockup__subtitle span",  # Primary selector
                ".job-card-container__company-name",      # Fallback 1
                ".jobs-unified-top-card__company-name a"  # Fallback 2
            ]:
                try:
                    company = base_element.find_element(By.CSS_SELECTOR, selector).text.strip()
                    break
                except:
                    continue

            # Location - multiple selector options
            location = ""
            for selector in [
                ".job-card-container__metadata-wrapper li:first-child",  # Primary selector
                ".job-card-container__metadata-item",                   # Fallback 1
                ".jobs-unified-top-card__primary-description span"      # Fallback 2
            ]:
                try:
                    location = base_element.find_element(By.CSS_SELECTOR, selector).text.strip()
                    break
                except:
                    continue

            # Salary - optional field
            salary = None
            for selector in [
                ".job-card-container__metadata-wrapper li:nth-child(2)",  # Primary selector
                ".job-salary",                                           # Fallback 1
                ".job-card-container__salary-info"                        # Fallback 2
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


    def search(self, search_term: str):
        url = f"{self.base_url}search/?keywords={urllib.parse.quote(search_term)}"
        
        try:
            self.driver.get(url)
            print("Page loaded successfully")
        except Exception as e:
            print(f"Failed to load page: {e}")
            return []

        # Wait for page to load completely
        time.sleep(random.uniform(3, 6))

        # Check for login wall
        try:
            if "authwall" in self.driver.current_url:
                print("Hit a login wall, trying to login again")
                actions.login(self.driver, email, password)
                time.sleep(random.uniform(2, 4))
                self.driver.get(url)
                time.sleep(random.uniform(3, 5))
        except:
            pass

        # Scroll to load more jobs
        # for _ in range(10):
        #     self.scroll_to_bottom()
        #     time.sleep(random.uniform(1, 2))
        scroll_button_value = ""
        try:
            # Wait for the buttons with specific text to appear
            jump_buttons = WebDriverWait(self.driver, 10).until(
                EC.presence_of_all_elements_located((By.XPATH,
                    "//button[contains(., 'Jump to active job details') or contains(., 'Jump to active search result')]"))
            )

            # Once we have the button(s), traverse up to their common ancestor <div>
            for button in jump_buttons:
                ancestor_div = button.find_element(By.XPATH, "./ancestor::div[1]")
                dynamic_class = ancestor_div.get_attribute("class")
                if dynamic_class:
                    print(f"[INFO] Found dynamic container class: {dynamic_class}")
                    scroll_button_value = dynamic_class
                    break
        except Exception as e:
            print(f"[ERROR] Could not find dynamic container: {e}")
            return []

        #self.scroll_to_bottom(pause_time=2, max_attempts=30, min_job_count=25)
        if scroll_button_value:
            self.scroll_to_bottom(
                pause_time=2,
                max_attempts=10,
                min_job_count=25,
                scroll_button_value=scroll_button_value
            )
        else:
            print("[ERROR] Cannot scroll, scroll_button_value not found")
            return []


        #Find job listings container with multiple fallbacks
        container = None
        container_selectors = [
            "div.scaffold-layout__list-container",  # Primary selector
            "div.jobs-search-results-list",         # Fallback 1
            "div.jobs-search-results",              # Fallback 2
            "div.scaffold-layout__list",            # Fallback 3
            "main.scaffold-layout__main"            # Fallback 4
        ]

        for selector in container_selectors:
            try:
                container = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
                print(f"Found container with selector: {selector}")
                break
            except:
                continue

        if not container:
            print("Could not find job listings container. Current page source:")
            print(self.driver.page_source[:2000])  # Print first 2000 chars for debugging
            return []

        # Find all job cards with multiple selectors
        job_cards = []
        for selector in [
            "li.jobs-search-results__list-item",  # Primary selector
            "li.scaffold-layout__list-item",       # Fallback 1
            "div.job-card-container",              # Fallback 2
            "section.jobs-search-results__list-item" # Fallback 3
        ]:
            try:
                job_cards = container.find_elements(By.CSS_SELECTOR, selector)
                if job_cards:
                    print(f"Found {len(job_cards)} jobs with selector: {selector}")
                    break
            except:
                continue

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
    # def search(self, search_term: str):
    #     url = f"{self.base_url}search/?keywords={urllib.parse.quote(search_term)}"
        
    #     try:
    #         self.driver.get(url)
    #         print("Page loaded successfully")
    #     except Exception as e:
    #         print(f"Failed to load page: {e}")
    #         return []

    #     # Wait for page to load completely
    #     time.sleep(random.uniform(3, 6))

    #     # Check for login wall and login if necessary
    #     try:
    #         if "authwall" in self.driver.current_url:
    #             print("Hit a login wall, trying to login again")
    #             actions.login(self.driver, email, password)
    #             time.sleep(random.uniform(2, 4))
    #             self.driver.get(url)
    #             time.sleep(random.uniform(3, 5))
    #     except:
    #         pass

    #     # Scroll to load more jobs
    #     self.scroll_to_bottom(pause_time=2, max_attempts=30)

    #     # Find the job listings container
    #     container = None
    #     container_selectors = [
    #         "div.scaffold-layout__list-container",  # Primary selector
    #         "div.jobs-search-results-list",         # Fallback 1
    #         "div.jobs-search-results",              # Fallback 2
    #         "div.scaffold-layout__list",            # Fallback 3
    #         "main.scaffold-layout__main"            # Fallback 4
    #     ]

    #     for selector in container_selectors:
    #         try:
    #             container = WebDriverWait(self.driver, 10).until(
    #                 EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
    #             print(f"Found container with selector: {selector}")
    #             break
    #         except:
    #             continue

    #     if not container:
    #         print("Could not find job listings container. Current page source:")
    #         print(self.driver.page_source[:2000])  # Print first 2000 chars for debugging
    #         return []

    #     # Find all job cards and process them
    #     job_cards = []
    #     for selector in [
    #         "li.jobs-search-results__list-item",  # Primary selector
    #         "li.scaffold-layout__list-item",       # Fallback 1
    #         "div.job-card-container",              # Fallback 2
    #         "section.jobs-search-results__list-item" # Fallback 3
    #     ]:
    #         try:
    #             job_cards = container.find_elements(By.CSS_SELECTOR, selector)
    #             if job_cards:
    #                 print(f"Found {len(job_cards)} jobs with selector: {selector}")
    #                 break
    #         except:
    #             continue

    #     job_results = []
    #     for card in job_cards:
    #         try:
    #             job = self.scrape_job_card(card)
    #             if job:
    #                 job_results.append(job)
    #         except Exception as e:
    #             print(f"Failed to process a job card: {e}")
    #             continue

    #     return job_results


    


try:
    # Login with random delay
    print("Logging in...")
    actions.login(driver, email, password)
    time.sleep(random.uniform(3, 5))

    # Initialize job search
    job_search = FixedJobSearch(driver=driver, close_on_complete=False, scrape=False)
    
    # Perform search with error handling
    print("Starting search...")
    job_listings = job_search.search("Machine Learning Engineer")
    
    # Print results
    print(f"\nFound {len(job_listings)} jobs:")
    for i, job in enumerate(job_listings, 1):
        print(f"\n{i}. {job.job_title} at {job.company}")
        print(f"   Location: {job.location}")
        if hasattr(job, 'salary') and job.salary:
            print(f"   Salary: {job.salary}")
        print(f"   URL: {job.linkedin_url}")

except Exception as e:
    print(f"\nCritical error: {str(e)}")
    # Take screenshot for debugging
    driver.save_screenshot("error_screenshot.png")
    print("Saved screenshot as error_screenshot.png")

finally:
    # Clean up
    driver.quit()
    print("\nBrowser closed.")

###############################################################################################################
###############################################################################################################