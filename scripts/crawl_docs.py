import requests
from bs4 import BeautifulSoup
import os
import json
from urllib.parse import urljoin

class LangChainDocCrawler:
    """LangChain官方文档爬取器"""
    
    def __init__(self, base_url: str = "https://python.langchain.com/"):
        self.base_url = base_url
        self.output_dir = "./data/raw"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def crawl(self, start_url: str, max_pages: int = 100):
        """爬取文档"""
        visited = set()
        to_visit = [start_url]
        pages_crawled = 0
        
        while to_visit and pages_crawled < max_pages:
            url = to_visit.pop(0)
            
            if url in visited:
                continue
            
            try:
                content = self._fetch_page(url)
                if content:
                    self._save_page(url, content)
                    visited.add(url)
                    pages_crawled += 1
                    print(f"已爬取 {pages_crawled}/{max_pages}: {url}")
                    
                    # 提取新链接
                    links = self._extract_links(content)
                    for link in links:
                        if link not in visited and self._is_doc_page(link):
                            to_visit.append(link)
            
            except Exception as e:
                print(f"爬取失败 {url}: {e}")
    
    def _fetch_page(self, url: str) -> str:
        """获取页面内容"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; DocCrawler/1.0)'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.text
    
    def _extract_links(self, html: str) -> list:
        """提取页面链接"""
        soup = BeautifulSoup(html, 'html.parser')
        links = []
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            if href.startswith('/'):
                href = urljoin(self.base_url, href)
            links.append(href)
        return links
    
    def _is_doc_page(self, url: str) -> bool:
        """判断是否为文档页面"""
        return 'langchain.com' in url and not any(
            ext in url for ext in ['.pdf', '.jpg', '.png', '.zip']
        )
    
    def _save_page(self, url: str, content: str):
        """保存页面"""
        filename = url.replace('/', '_').replace(':', '_') + '.html'
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

if __name__ == "__main__":
    crawler = LangChainDocCrawler()
    crawler.crawl(
        start_url="https://python.langchain.com/docs/get_started/introduction",
        max_pages=50
    )