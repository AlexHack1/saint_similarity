import requests
from bs4 import BeautifulSoup
import pandas as pd

url = "https://en.wikipedia.org/wiki/List_of_Catholic_saints"
headers = {'User-Agent': 'SaintSimilarityProject/1.0 (yourname@example.com)'}

response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')

all_saints = []

# The saints are in tables with the class 'wikitable'
tables = soup.find_all('table', {'class': 'wikitable'})

for table in tables:
    rows = table.find_all('tr')[1:]  # Skip the header row
    for row in rows:
        cells = row.find_all('td')
        if not cells:
            continue
            
        # The name is usually in the first column
        name_cell = cells[0]
        link_tag = name_cell.find('a')
        
        if link_tag and 'href' in link_tag.attrs:
            name = link_tag.get_text()
            # Construct the full Wikipedia URL
            link = "https://en.wikipedia.org" + link_tag['href']
            # Get the 'title' used for the API (e.g., Saint_Stephen)
            wiki_title = link_tag['href'].replace('/wiki/', '')
            
            all_saints.append({
                'name': name,
                'wiki_title': wiki_title,
                'url': link
            })

# Convert to DataFrame
df_links = pd.DataFrame(all_saints)
print(f"Found {len(df_links)} saints with links.")
#print(df_links[['name','url']].head())

import wikipediaapi
import time

# 1. Initialize the API
# Again, use a clear user-agent to avoid 403s
wiki = wikipediaapi.Wikipedia(
    user_agent="SaintSimilarityProject/1.0 alexanderjwh@gmail.com",
    language='en',
    extract_format=wikipediaapi.ExtractFormat.WIKI
)

def fetch_content(title):
    try:
        page = wiki.page(title)
        if page.exists():
            return page.text
        return None
    except Exception as e:
        print(f"Error fetching {title}: {e}")
        return None

# 2. Fetch the text (Starting with a slice of 50 to test)
print("Fetching biographies... this may take a few minutes.")
df_links['biography'] = df_links['wiki_title'].apply(fetch_content)

# 3. Drop any that failed to load
df_links = df_links.dropna(subset=['biography'])

# 4. Save locally so you never have to do this again!
df_links.to_csv('saints_data_full.csv', index=False)
print(f"Saved {len(df_links)} biographies to CSV.")