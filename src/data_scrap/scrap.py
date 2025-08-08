import os
import asyncio
# from data_scrap.urls import KNOWLEDGE_BASE_URLS
from urls.links_unique import ALL_URLS
from data_scrap.data_scrap import scrape_multiple_urls

# ✅ Create directory to store scraped docs
OUTPUT_DIR = "scraped_docs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ✅ 70 URLs (replace with your full list)
url_map = ALL_URLS

async def run_scraper():
    print(f"🔍 Starting scrape for {len(url_map)} URLs...")
    results = await scrape_multiple_urls(url_map)

    for result in results:
        file_path = os.path.join(OUTPUT_DIR, f"{result['key']}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(result["full_text"])
        print(f"📄 Saved: {file_path}")

    print("✅ Scraping complete.")

if __name__ == "__main__":
    asyncio.run(run_scraper())
