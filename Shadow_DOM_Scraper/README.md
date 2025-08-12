# Shadow DOM Web Scraper

A Python-based web scraping tool that can extract content from modern web applications using Shadow DOM technology, built with Selenium WebDriver.

## Author
[sebieire](https://github.com/sebieire/)

## Year
2023 (extraction & revision 2025)

## Overview
This project provides a specialized web scraper capable of extracting content from websites that use Shadow DOM - a web standard that encapsulates parts of a web page to keep them separate from the main document. Traditional web scrapers often fail to access content within Shadow DOM boundaries, making this tool essential for scraping modern web applications.

This is originally a component of a larger piece of software and has been "extracted" (standalone) for portfolio purposes only.

## What is Shadow DOM?
Shadow DOM is a web standard that allows hidden DOM trees to be attached to elements in the regular DOM tree. It's commonly used in modern web frameworks and web components to encapsulate functionality and styling. Learn more about [Shadow DOM on MDN](https://developer.mozilla.org/en-US/docs/Web/API/Web_components/Using_shadow_DOM).

## Key Features
- **Shadow DOM Detection**: Automatically detects and extracts content from shadow root elements
- **Recursive DOM Traversal**: Traverses both regular DOM and shadow DOM trees recursively
- **Shadow Host Detection**: Identifies all elements hosting shadow roots on a page
- **Comparison Mode**: Compare extraction results with and without shadow DOM handling
- **Headless Browser Support**: Can run with or without displaying browser window
- **Configurable Scrolling**: Simulates user scrolling to trigger lazy-loaded content

## Technical Implementation
- **Selenium WebDriver**: Browser automation for dynamic content rendering
- **Chrome DevTools Protocol**: Direct access to shadow roots via JavaScript execution
- **BeautifulSoup**: HTML parsing and analysis of extracted content
- **Context Manager Support**: Clean resource management with automatic browser cleanup

## Requirements
```
selenium==4.15.2
beautifulsoup4==4.12.2
webdriver-manager==4.0.1
```

## Installation
```bash
# Clone the repository
git clone <repository-url>

# Navigate to project directory
cd Shadow_DOM_Scraper

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Extraction
```python
from shadow_dom_extractor import ShadowDOMExtractor

# Create extractor instance
with ShadowDOMExtractor(headless=True) as extractor:
    # Extract content including shadow DOM
    html, has_shadow = extractor.extract_with_shadow_dom("https://example.com")
    
    print(f"Shadow DOM detected: {has_shadow}")
    print(f"HTML length: {len(html)} characters")
```

### Find Shadow Hosts
```python
# Detect all shadow host elements on a page
shadow_hosts = extractor.find_shadow_hosts("https://example.com")
for host in shadow_hosts:
    print(f"Found shadow host: {host['tagName']} (id='{host['id']}')")
```

### Run Demo
```bash
python demo_shadow_dom.py
```

The demo includes:
- Basic extraction from multiple sites
- Shadow host detection and analysis
- Comparison between regular and shadow DOM extraction
- Custom website testing interface

## Project Structure
```
Shadow_DOM_Scraper/
├── shadow_dom_extractor.py   # Main extraction class
├── demo_shadow_dom.py         # Demonstration and examples
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── venv/                      # Virtual environment (not in repo)
└── gitignore/                 # Files not uploaded to git
    ├── test_shadow_dom.py     # Basic test file
    └── simple_test.py         # Simple test script
```

## How It Works
1. **Browser Initialization**: Sets up Chrome WebDriver with appropriate options
2. **Page Loading**: Navigates to target URL and waits for initial content
3. **Shadow Root Detection**: Uses JavaScript execution to find shadow roots
4. **Recursive Traversal**: Walks through both regular DOM and shadow DOM trees
5. **Content Reconstruction**: Rebuilds complete HTML including shadow content
6. **Cleanup**: Properly closes browser and releases resources

## Use Cases
- Scraping modern Single Page Applications (SPAs)
- Extracting content from web components
- Testing web applications that use Shadow DOM
- Analyzing YouTube, Chrome Extensions pages, and other Google properties
- Gathering data from polymer-based or lit-element websites

## Limitations
- Requires Chrome/Chromium browser
- Performance depends on page complexity and shadow DOM depth
- Some closed shadow roots may not be accessible
- Heavy sites like YouTube may take time to fully process

## Future Enhancements
- Support for other browsers (Firefox, Safari)
- Parallel extraction for multiple URLs
- Caching mechanism for repeated extractions
- Export to various formats (JSON, CSV)
- Shadow DOM modification capabilities