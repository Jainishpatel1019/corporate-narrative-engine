import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion.html_parser import SECHTMLParser

@pytest.fixture
def parser():
    return SECHTMLParser()

@pytest.fixture
def fake_html_full():
    return """
    <html>
        <body>
            <h1>Apple Inc. 10-K</h1>
            <div>
                <b>Item 1A. Risk Factors</b>
                <p>Investing in our company involves risks. We might not sell enough phones.</p>
            </div>
            <span>Item 1B. Unresolved Staff Comments</span>
            <p>None.</p>
            <div>
                <h2>Item 7. Management's Discussion and Analysis</h2>
                <p>We had a great year. Revenue was up 20%.</p>
            </div>
            <div>
                <b>Item 8. Financial Statements</b>
                <p>Here are the numbers.</p>
            </div>
            <div>
                <h3>Forward-Looking Statements</h3>
                <p>These statements are predictions and involve known and unknown risks.</p>
            </div>
            <div>Part III</div>
        </body>
    </html>
    """

@pytest.fixture
def fake_html_missing_risk():
    return """
    <html>
        <body>
            <h1>Microsoft 10-Q</h1>
            <div>
                <h2>Item 2. Management's Discussion and Analysis</h2>
                <p>Cloud revenue grew significantly.</p>
            </div>
            <div>
                <b>Item 3. Quantitative and Qualitative Disclosures</b>
                <p>Nothing much here.</p>
            </div>
        </body>
    </html>
    """

def test_parse_full_html(parser, fake_html_full):
    results = parser.parse(fake_html_full)
    
    # MD&A Assertions
    assert "MD&A" in results
    assert "Item 7." in results["MD&A"]
    assert "We had a great year." in results["MD&A"]
    assert "Item 8." not in results["MD&A"]
    
    # Risk Factors Assertions
    assert "Risk Factors" in results
    assert "Item 1A." in results["Risk Factors"]
    assert "sell enough phones" in results["Risk Factors"]
    assert "Item 1B." not in results["Risk Factors"]
    
    # Forward Looking Statements Assertions
    assert "Forward Looking Statements" in results
    assert "These statements are predictions" in results["Forward Looking Statements"]
    assert "Part III" not in results["Forward Looking Statements"]

def test_parse_missing_sections(parser, fake_html_missing_risk):
    results = parser.parse(fake_html_missing_risk)
    
    # MD&A Should be present (10-Q uses Item 2)
    assert "MD&A" in results
    assert results["MD&A"] != ""
    assert "Item 2." in results["MD&A"]
    assert "Cloud revenue grew" in results["MD&A"]
    
    # Risk factors should be empty string
    assert "Risk Factors" in results
    assert results["Risk Factors"] == ""
    
    # Forward looking statements also missing
    assert "Forward Looking Statements" in results
    assert results["Forward Looking Statements"] == ""

def test_stripped_html_tags(parser, fake_html_full):
    results = parser.parse(fake_html_full)
    
    for key, text in results.items():
        if text:
            # Verify there are no HTML tags
            assert "<p>" not in text
            assert "<b>" not in text
            assert "</div>" not in text
            assert "<h2>" not in text
            assert "<h3>" not in text

def test_no_exception_on_bad_html(parser):
    # Pass None or weird strings to ensure exceptions aren't raised
    results = parser.parse(None)
    assert all(value == "" for value in results.values())
    
    results = parser.parse("")
    assert all(value == "" for value in results.values())
