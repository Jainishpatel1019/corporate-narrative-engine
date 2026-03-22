import logging
import time
import requests
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class EDGARClient:
    BASE_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
    USER_AGENT = "Jainish Patel jainishpatel153@gmail.com"
    
    def __init__(self, requests_per_second: int = 10, max_retries: int = 3):
        self.requests_per_second = requests_per_second
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.USER_AGENT})
        self._last_request_time = 0.0

    def _wait_for_rate_limit(self):
        """Ensure we don't exceed the configured rate limit."""
        current_time = time.time()
        time_since_last_req = current_time - self._last_request_time
        min_interval = 1.0 / self.requests_per_second
        if time_since_last_req < min_interval:
            time.sleep(min_interval - time_since_last_req)
        self._last_request_time = time.time()

    def _fetch_with_retry(self, url: str) -> Dict[str, Any]:
        """Fetch URL with exponential backoff for 429s and 5xx."""
        backoff = 1.0
        for attempt in range(self.max_retries + 1):
            self._wait_for_rate_limit()
            response = self.session.get(url)
            
            if response.status_code == 200:
                return response.json()
                
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == self.max_retries:
                    response.raise_for_status()
                logger.warning(f"Request failed with {response.status_code}. Retrying in {backoff} seconds...")
                time.sleep(backoff)
                backoff *= 2
            else:
                response.raise_for_status()
                
        # Fallback if loop finishes unexpectedly
        raise requests.exceptions.HTTPError(f"Failed after {self.max_retries} retries.")

    def get_filings(self, cik: str, form_types: List[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Fetches filings for a given CIK and filters by form_type.
        
        Returns a list of dicts with keys: 
        filing_id, cik, form_type, period_of_report, filed_date, accession_number
        """
        if form_types is None:
            form_types = ['10-K', '10-Q']
            
        # Ensure CIK is zero-padded to 10 digits as required by SEC API
        cik_padded = str(cik).zfill(10)
        url = self.BASE_URL.format(cik=cik_padded)
        
        data = self._fetch_with_retry(url)
        recent_filings = data.get("filings", {}).get("recent", {})
        
        if not recent_filings:
            return []
            
        accession_numbers = recent_filings.get("accessionNumber", [])
        forms = recent_filings.get("form", [])
        report_dates = recent_filings.get("reportDate", [])
        filing_dates = recent_filings.get("filingDate", [])
        
        results = []
        # SEC returns data in column format where each key maps to an array of rows
        for i in range(len(accession_numbers)):
            form = forms[i]
            if form in form_types:
                acc_num = accession_numbers[i]
                # Generate a filing_id by removing hyphens from the accession number
                filing_id = str(acc_num).replace("-", "")
                
                filing = {
                    "filing_id": filing_id,
                    "cik": str(cik),
                    "form_type": str(form),
                    "period_of_report": str(report_dates[i]),
                    "filed_date": str(filing_dates[i]),
                    "accession_number": str(acc_num)
                }
                results.append(filing)
                
                if len(results) >= limit:
                    break
                    
        return results
