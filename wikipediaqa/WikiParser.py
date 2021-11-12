import warnings
import wikipedia as wiki
import aiowiki

from bs4 import BeautifulSoup
import requests

import asyncio
import aiohttp

import sys

from .WikiParser import *

if 'ipykernel' in sys.modules:
    # workaround ipython notebooks already using async
    import nest_asyncio
    nest_asyncio.apply()


class UnOptimized(RuntimeWarning):
    pass


warnings.simplefilter('once', UnOptimized)


class WikiParser:
    """
    Class that handles parsing from wiki
    """
    def __init__(self, lang : str ="en"):
        """
        lang: language of wikipedia
        default "en"
        list of all languages can be foun here https://meta.wikimedia.org/wiki/List_of_Wikipedias
        """
        self.headers = {'user-agent': 'my-app/0.0.1'}
        self.awiki = aiowiki.Wiki.wikipedia(lang)
        wiki.set_lang(lang)
    
    def page(self, name):
        page = self.awiki.get_page(name)
        return Page(page, self)

    def search(self, text):
        return [self.page(i) for i in wiki.search(text)]

    def search_summary(self, text):
        summarys = asyncio.run(self._search_summary(text))
        return summarys

    async def _search_summary(self, text):
        pages = await awiki.opensearch(text)
        summarys = [await i.summary() for i in pages]
        return summarys

    def asearch(self, text):
        return asyncio.run(self._asearch(text))

    async def _asearch(self, text):
        search = await self.awiki.opensearch(text)
        return [Page(i) for i in search]

    def postprocess_text(self, text):
        """
        History[edit] -> History
        Python[231] -> Python
        """
        return re.sub(r'\[[^ ]*\]', '', text)
        
    def getText(self, url):
        """
        Gets text from page
        Returns list of paragraphs
        Paragraphs made with this rules:
            header + text below
            text + list below
            text without headers or lists near
        """
        response = requests.get(url, headers = self.headers)
        soup = BeautifulSoup(response.text)
        div = soup.find('div', class_ = "mw-parser-output").find_all(['p', 'h2', 'h3', 'ul', 'dl'])
        
        div = [i for i in div
               if (i.parent.has_attr('class')) and ('mw-parser-output' in i.parent['class'])]
        
        newdiv = [div[0].text]
        for i in range(1, len(div)):
            divnow, divprev = div[i], div[i-1]
            if divnow.name == 'p' and (divprev.name == 'h2' or divprev.name == 'h3'):
                newdiv.append(divprev.text + '\n' + divnow.text)
                continue

            elif ((divprev.name == 'p' or divprev.name == 'h2' or divprev.name == 'h3') and 
                    (divnow.name == 'ul' or divnow.name == 'dl')):

                newdiv.append(divnow.text + divprev.text)
                continue

            elif divnow.name == 'p':
                newdiv.append(divnow.text)
            
            if ((divnow.name == 'h2' or divnow.name == 'h3') and 
                         ('See also' in divnow.text or 'References' in divnow.text)):
                break

        return [self.postprocess_text(i) for i in newdiv if len(i) > 5]
    
    def getInfo(self, url):
        """
        Get infoBox from page
        """
        response = requests.get(url, headers = self.headers)
        soup = BeautifulSoup(response.text)

        tbody = soup.find('tbody')
        
        if not tbody:
            return ''
        
        tbody = tbody.find_all('tr')

        out = ''
        for i in tbody:
            try:
                tbody_th = i.find('th')
                tbody_th = tbody_th.text.replace('\n', '')
                tbody_td = i.find('td')
                tbody_td = tbody_td.text.replace('\n', '')
                out += f' {tbody_th} — {tbody_td}; \n'
            except: pass

        return out


class Page:
    def __init__(self, page, parser=WikiParser()):
        self.page = page
        self.title = page.title
        self.parser = parser

        self.url = None

        self.repr_count = 0

    def __repr__(self):
        warnings.warn(
            "Warning while displaying Page\n"
            "getting url of page takes a bit of time\n"
            "if you need to display many pages it's advised to use page.title",
             UnOptimized)

        return f"<Page {self.title}  {self.url()}>"

    def url(self):
        url = asyncio.run(self._getUrl())
        return url.view
    
    async def _getUrl(self):
        url = await self.page.urls()
        return url

    def text(self):
        url = self.url()
        text = self.parser.getText(url)
        return text
    
    def infoBox(self):
        url = self.url()
        info = self.parser.getInfo(url)
        return info

    def summary(self):
        try: summary = self.title + '\n' + asyncio.run(self._getSummary())
        except: summary = self.title + '\n' + self.text()[0]
        return summary
    
    async def _getSummary(self):
        summary = await self.page.summary()
        return summary
