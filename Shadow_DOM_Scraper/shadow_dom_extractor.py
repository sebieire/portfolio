#!/usr/bin/env python3
"""
Shadow DOM Extractor - Extracts content from shadow root elements
Author: https://github.com/sebieire/
Date: 2023
"""

import time
import random
from typing import List, Tuple, Optional, Dict
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


class ShadowDOMExtractor:
    """Extracts content from web pages with Shadow DOM elements."""
    
    def __init__(self, headless: bool = True, wait_timeout: int = 10):
        self.headless = headless
        self.wait_timeout = wait_timeout
        self.driver = None
        self.shadow_root_found = False
        self.temp_html = ""
        self.css_selector = "*"  # Default to all elements
        
    def setup_driver(self) -> webdriver.Chrome:
        """Set up Chrome WebDriver with options."""
        options = Options()
        if self.headless:
            options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=options)
        return self.driver
    
    def extract_shadow_dom_children(self, element: WebElement) -> Tuple[List[WebElement], bool]:
        """
        Looks for Shadow Root and returns all immediate children elements.
        NOTE: only one shadow root can be created per element (as of 2023)
        """
        shadow_root_present = False
        shadow_children = []
        
        # (apparently) only one shadow root can be created per element
        shadow_root = self.driver.execute_script('return arguments[0].shadowRoot', element)
        
        if shadow_root:
            shadow_root_present = True
            
            # this will find all possible children (with all nested elements) within the shadow root
            all_shadow_elements = shadow_root.find_elements(By.CSS_SELECTOR, self.css_selector)
            
            for shadow_element in all_shadow_elements:
                # find out parent node of each 'shadow_element'
                parent_node = self.driver.execute_script('return arguments[0].parentNode', shadow_element)
                
                # only append immediate children of the shadowRoot element
                if isinstance(parent_node, webdriver.remote.shadowroot.ShadowRoot):
                    shadow_children.append(shadow_element)
        
        return shadow_children, shadow_root_present
    
    def get_element_tag_info(self, element: WebElement) -> Tuple[str, str]:
        """Returns html 'tag_head' (<div class=...>) and 'tag_name_only' (div)."""
        outer_html = element.get_attribute('outerHTML')
        if not outer_html:
            return "", ""
            
        # Get the opening tag
        tag_head = outer_html.split('>', 1)[0] + '>'
        
        # Extract tag name
        tag_name = tag_head.split(' ', 1)[0]
        if tag_name.startswith('<'):
            tag_name = tag_name[1:]
        if tag_name.endswith('>'):
            tag_name = tag_name[:-1]
            
        return tag_head, tag_name
    
    def traverse_dom_with_shadow(self, element: WebElement, depth: int = 0) -> str:
        """Recursive function. Walks and reconstructs all siblings & children for given element."""
        # get all children (including potential shadow root children)
        shadow_children, has_shadow = self.extract_shadow_dom_children(element)
        if has_shadow:
            self.shadow_root_found = True
            
        regular_children = self.driver.execute_script('return arguments[0].children', element)
        
        # combine all children
        all_children = list(shadow_children) + list(regular_children)
        
        # no children
        if not all_children:
            inner_html = element.get_attribute('innerHTML') or ""
            self.temp_html += inner_html
        else:
            # Process each child
            for child in all_children:
                tag_head, tag_name = self.get_element_tag_info(child)
                
                if tag_head:
                    self.temp_html += tag_head
                    
                # Recursive call for child element
                self.traverse_dom_with_shadow(child, depth + 1)
                
                if tag_name:
                    self.temp_html += f"</{tag_name}>"
                    
        return self.temp_html
    
    def extract_with_shadow_dom(self, url: str, scroll: bool = True) -> Tuple[str, bool]:
        """Wrapper for extraction. Returns full HTML page and bool if shadow elements found."""
        if not self.driver:
            self.setup_driver()
            
        try:
            # Reset state
            self.shadow_root_found = False
            self.temp_html = ""
            
            # Navigate to URL
            self.driver.get(url)
            
            # Wait for initial load
            WebDriverWait(self.driver, self.wait_timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Optional scrolling to trigger lazy loading
            if scroll:
                self._scroll_page()
            
            # Start reconstruction
            html_result = "<html>"
            
            # Get head element
            head = self.driver.find_element(By.TAG_NAME, 'head')
            html_result += head.get_attribute('outerHTML')
            
            # Get body and traverse
            body = self.driver.find_element(By.TAG_NAME, 'body')
            body_tag, _ = self.get_element_tag_info(body)
            html_result += body_tag
            
            # Traverse the body including shadow DOM
            body_content = self.traverse_dom_with_shadow(body)
            html_result += body_content
            
            html_result += "</body></html>"
            
            return html_result, self.shadow_root_found
            
        except Exception as e:
            print(f"Error extracting from {url}: {str(e)}")
            return "", False
    
    def _scroll_page(self, scrolls: int = 3):
        """Simulates browser scrolling."""
        for i in range(scrolls):
            scroll_height = random.randint(300, 700)
            self.driver.execute_script(f"window.scrollBy(0, {scroll_height})")
            time.sleep(random.uniform(0.5, 1.5))
    
    def find_shadow_hosts(self, url: str) -> List[Dict]:
        """Find all elements hosting shadow roots."""
        if not self.driver:
            self.setup_driver()
            
        self.driver.get(url)
        time.sleep(2)  # Allow page to load
        
        # JavaScript to find all shadow hosts
        script = """
        const shadows = [];
        const walker = document.createTreeWalker(
            document.body,
            NodeFilter.SHOW_ELEMENT,
            {
                acceptNode: function(node) {
                    return node.shadowRoot ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_SKIP;
                }
            }
        );
        
        let node;
        while(node = walker.nextNode()) {
            shadows.push({
                tagName: node.tagName,
                id: node.id,
                className: node.className,
                shadowRootMode: node.shadowRoot.mode
            });
        }
        return shadows;
        """
        
        shadow_hosts = self.driver.execute_script(script)
        return shadow_hosts
    
    def close(self):
        """Clean up and close the browser driver."""
        if self.driver:
            self.driver.quit()
            self.driver = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Example usage and demonstration
if __name__ == "__main__":
    # example using the shadow DOM extractor
    extractor = ShadowDOMExtractor(headless=True) # headless settings
    
    # Test URL (you can replace with any URL that uses Shadow DOM)
    # test_url = "https://www.example.com"
    test_url = "https://shop.polymer-project.org/"  # example with known shadow DOM usage
    
    
    print("Starting Shadow DOM extraction...")
    print(f"Target URL: {test_url}")
    print("-" * 50)
    
    try:
        # Extract content including shadow DOM
        html_content, has_shadow = extractor.extract_with_shadow_dom(test_url)
        
        print(f"Extraction complete!")
        print(f"Shadow DOM detected: {has_shadow}")
        print(f"HTML length: {len(html_content)} characters")
        
        # Find shadow hosts
        print("\nSearching for shadow hosts...")
        shadow_hosts = extractor.find_shadow_hosts(test_url)
        
        if shadow_hosts:
            print(f"Found {len(shadow_hosts)} shadow host(s):")
            for host in shadow_hosts[:5]:  # Show first 5
                print(f"  - {host['tagName']} (id='{host['id']}', class='{host['className']}')")
        else:
            print("No shadow hosts found on this page")
            
    finally:
        extractor.close()
        print("\nExtractor closed.")