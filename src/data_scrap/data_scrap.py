import asyncio
import httpx

from typing import Dict, Any, List, Set
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from datetime import datetime
import hashlib
import re
from webdriver_manager.chrome import ChromeDriverManager


async def is_connected(test_url: str = "https://www.google.com", timeout: int = 5) -> bool:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(test_url)
            return response.status_code == 200
    except httpx.RequestError:
        return False


def hash_text(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()


def extract_text_from_body(soup, seen_hashes: Set[str]) -> str:
    def clean_text(text):
        return re.sub(r'\s+', ' ', text.strip())

    text_blocks = []
    body = soup.find("body")
    if body:
        for tag in body.find_all(True):  # All tags inside <body>
            tag_text = clean_text(tag.get_text(separator=" ", strip=True))
            if tag_text:
                text_hash = hash_text(tag_text)
                if text_hash not in seen_hashes:
                    seen_hashes.add(text_hash)
                    text_blocks.append(tag_text)

    return "\n".join(text_blocks)


async def scrape_url(key: str, url: str, retries: int = 3) -> Dict[str, Any]:
    if not await is_connected():
        log_error(f"No internet connection. Cannot scrape: {url}")
        return {}

    attempt = 0
    while attempt < retries:
        try:
            # Headless Chrome setup
            chrome_options = Options()
            chrome_options.add_argument("--headless=new")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")

            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

            try:
                driver.get(url)

                # Wait for body to be loaded
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )

                # Scroll once (avoid loading footer/header multiple times)
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                await asyncio.sleep(3)

                # Get page source after JavaScript execution
                page_source = driver.page_source
                soup = BeautifulSoup(page_source, "html.parser")

                # Remove noisy tags
                for tag in soup(["script", "style", "noscript", "iframe", "svg", "header", "footer", "aside"]):
                    tag.decompose()

                seen_hashes = set()
                full_text = extract_text_from_body(soup, seen_hashes)

                print(f"✅ Scraped: {url}")
                return {
                    "key": key,
                    "url": url,
                    "full_text": full_text,
                }

            finally:
                driver.quit()

        except Exception as e:
            attempt += 1
            log_error(f"Error scraping {url} on attempt {attempt}: {e}")
            if attempt >= retries:
                return {}


def log_error(message: str, log_file: str = "scraper_errors.log"):
    timestamp = datetime.now().isoformat()
    with open(log_file, "a") as f:
        f.write(f"[{timestamp}] {message}\n")
    print(f"❌ {message}")


async def scrape_multiple_urls(url_map: Dict[str, str]) -> List[Dict[str, Any]]:
    seen_urls = set()
    tasks = []

    for key, url in url_map.items():
        if url in seen_urls:
            print(f"⏩ Skipping duplicate URL in input: {url}")
            continue
        seen_urls.add(url)
        tasks.append(scrape_url(key, url))

    results = await asyncio.gather(*tasks)
    return [doc for doc in results if doc]


