import requests
from bs4 import BeautifulSoup
import os
import re
import sys
from urllib.parse import urljoin

def download_paper(paper_url):
    # Extract paper ID from URL (e.g., '2512.02953' from 'https://arxiv.org/abs/2512.02953')
    paper_id = paper_url.rstrip('/').split('/')[-1]

    # Validate paper ID format (should match pattern like 2512.02953 or similar)
    # arXiv IDs are typically in format: YYMM.NNNNN or YYMM.NNNNN vN
    if not re.match(r'^\d{4}\.\d{4,5}(v\d+)?$', paper_id):
        paper_id = None

    # Send a request to the paper URL
    response = requests.get(paper_url)

    if response.status_code != 200:
        print("Failed to retrieve the page")
        return

    # Parse the HTML content of the page
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the title of the paper
    title_tag = soup.find('h1', class_='title')
    if not title_tag:
        print("Title not found")
        return



    # Clean up the title to make it filename-friendly
    # Remove "Title:" prefix if present, then clean special characters
    title = title_tag.get_text().strip()
    if title.startswith('Title:'):
        title = title[6:].strip()
    title = title.replace("\n", " ").replace(":", "").replace("/", "_").replace(",", "")


    # Prefix the paper ID to the title only if it's valid
    if paper_id:
        title = f"{paper_id}_{title}"

    # Find the PDF download link. Try multiple strategies because arXiv
    # doesn't always set the same attributes on the link.
    # 1) look for an anchor with title='View PDF'
    pdf_link_tag = soup.find('a', title='View PDF')
    # 2) fallback: anchor whose visible text contains 'View PDF'
    if not pdf_link_tag:
        pdf_link_tag = soup.find('a', string=lambda s: s and 'View PDF' in s)
    # 3) fallback: anchor whose href contains '/pdf/' (most reliable)
    if not pdf_link_tag:
        pdf_link_tag = soup.find('a', href=lambda h: h and '/pdf/' in h)
    if not pdf_link_tag:
        print("PDF link not found")
        return

    pdf_url = urljoin(paper_url, pdf_link_tag['href'])

    # Send a GET request to download the PDF
    pdf_response = requests.get(pdf_url)

    if pdf_response.status_code == 200:
        # Save the PDF with the title as the filename
        pdf_filename = f"{title}.pdf"

        # Clean up filename to avoid invalid characters
        pdf_filename = pdf_filename.replace(" ", "_").replace(":", "").replace("/", "_").replace(",", "")

        # Save to the current directory (or specify your directory)
        with open(pdf_filename, 'wb') as f:
            f.write(pdf_response.content)

        print(f"Downloaded: {pdf_filename}")
    else:
        print("Failed to download the PDF")

# Example usage:
if __name__ == "__main__":
    # Check if paper URL is provided as command-line argument
    if len(sys.argv) > 1:
        paper_url = sys.argv[1]
    else:
        # Prompt user for paper URL
        paper_url = input("Enter the arXiv paper URL (e.g., https://arxiv.org/abs/2512.02953): ").strip()

    if not paper_url:
        print("Error: Paper URL cannot be empty")
        sys.exit(1)

    download_paper(paper_url)
