import re
from bs4 import BeautifulSoup
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class SECHTMLParser:
    def __init__(self):
        # Use (?im) for case-insensitive and multi-line matching
        # Lookahead (?= ... ) stops at the next major SEC item or part
        self.stop_pattern = r"(?:^item\s+[0-9a-z]+(?:[\.\s]|$)|^part\s+[ivx]+(?:[\.\s]|$)|\Z)"
        
        self.patterns = {
            "MD&A": re.compile(
                r"(?im)^item\s+(?:7|2)\.?\s+management.*?discussion.*?(?=" + self.stop_pattern + r")", 
                re.DOTALL
            ),
            "Risk Factors": re.compile(
                r"(?im)^item\s+1a\.?\s+risk\s+factors.*?(?=" + self.stop_pattern + r")", 
                re.DOTALL
            ),
            "Forward Looking Statements": re.compile(
                r"(?im)^(?:cautionary\s+note\s+regarding\s+)?forward[\-\s]looking\s+statements.*?(?=" + self.stop_pattern + r")", 
                re.DOTALL
            )
        }

    def parse(self, html: str) -> Dict[str, str]:
        results = {
            "MD&A": "",
            "Risk Factors": "",
            "Forward Looking Statements": ""
        }
        
        if not html or not isinstance(html, str) or not html.strip():
            return results
            
        try:
            soup = BeautifulSoup(html, "html.parser")
            
            # Strip all script and style elements
            for script in soup(["script", "style"]):
                script.extract()
                
            # Get raw text, stripping HTML
            text = soup.get_text(separator='\n')
            
            # Clean up into normalized lines to make regex matching robust
            lines = [line.strip() for line in text.split('\n')]
            clean_text = '\n'.join([line for line in lines if line])
            
            for section, pattern in self.patterns.items():
                match = pattern.search(clean_text)
                if match:
                    results[section] = match.group(0).strip()
                    
        except Exception as e:
            logger.error(f"Error parsing HTML: {e}")
            # Graceful degradation - never raise exception
            pass
            
        return results
