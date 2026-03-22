import os
import sys
import pytest
import requests
from unittest.mock import patch, MagicMock

# Ensure we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion.edgar_client import EDGARClient

@pytest.fixture
def mock_sec_response():
    return {
        "cik": "320193",
        "entityType": "operating",
        "filings": {
            "recent": {
                "accessionNumber": [
                    "0000320193-23-000106",
                    "0000320193-23-000077",
                    "0000320193-23-000064",
                    "0000320193-22-000108"
                ],
                "filingDate": [
                    "2023-11-03",
                    "2023-08-04",
                    "2023-05-05",
                    "2022-10-28"
                ],
                "reportDate": [
                    "2023-09-30",
                    "2023-07-01",
                    "2023-04-01",
                    "2022-09-24"
                ],
                "form": [
                    "10-K",
                    "10-Q",
                    "8-K",
                    "10-K"
                ]
            }
        }
    }

def test_get_filings_schema_and_filtering(mock_sec_response):
    client = EDGARClient()
    
    with patch('requests.Session.get') as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_sec_response
        mock_get.return_value = mock_resp
        
        # Test 10-K and 10-Q (should filter out the 8-K)
        filings = client.get_filings('320193', form_types=['10-K', '10-Q'], limit=10)
        
        assert len(filings) == 3 
        
        # Check required keys in return schema
        for filing in filings:
            assert "filing_id" in filing
            assert "cik" in filing
            assert "form_type" in filing
            assert "period_of_report" in filing
            assert "filed_date" in filing
            assert "accession_number" in filing
            
        # Form type filtering validation
        forms = [f["form_type"] for f in filings]
        assert "10-K" in forms
        assert "10-Q" in forms
        assert "8-K" not in forms
        
        # Verify limit argument behaves correctly
        filings_limited = client.get_filings('320193', form_types=['10-K'], limit=1)
        assert len(filings_limited) == 1
        assert filings_limited[0]["form_type"] == "10-K"

@patch('time.sleep', return_value=None) # Mock sleep to speed up the tests
def test_retry_on_429(mock_sleep, mock_sec_response):
    client = EDGARClient(max_retries=2)
    
    with patch('requests.Session.get') as mock_get:
        # Simulate two 429 errors followed by a successful 200 response
        resp_429 = MagicMock()
        resp_429.status_code = 429
        
        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.json.return_value = mock_sec_response
        
        mock_get.side_effect = [resp_429, resp_429, resp_200]
        
        filings = client.get_filings('320193')
        
        assert len(filings) > 0
        assert mock_get.call_count == 3
        
@patch('time.sleep', return_value=None)
def test_retry_exhaustion_raises_error(mock_sleep):
    client = EDGARClient(max_retries=1)
    
    with patch('requests.Session.get') as mock_get:
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.raise_for_status.side_effect = requests.exceptions.HTTPError("429 Too Many Requests")
        
        # Only set enough failures to exhaust retries
        mock_get.side_effect = [resp_429, resp_429]
        
        with pytest.raises(requests.exceptions.HTTPError):
            client.get_filings('320193')
            
        # Initial call (1) + Retry (1) = 2 attempts total
        assert mock_get.call_count == 2
