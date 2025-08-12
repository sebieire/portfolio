#!/usr/bin/env python3
"""
Demo script for Shadow DOM Extractor
"""

from shadow_dom_extractor import ShadowDOMExtractor
import json
from bs4 import BeautifulSoup
import time


def demo_basic_extraction():
    print("\n" + "="*60)
    print("DEMO 1: Basic Shadow DOM Extraction")
    print("="*60)
    
    # Test URLs
    test_urls = [
        "https://www.example.com", # no shadow DOM
        "https://shop.polymer-project.org/", # has shadow DOM
        "https://developer.mozilla.org/en-US/docs/Web/API/Web_components/Using_shadow_DOM", # no shadow DOM
        "https://lit.dev/docs/components/shadow-dom/", # has shadow DOM
    ]
    
    with ShadowDOMExtractor(headless=True) as extractor:
        for url in test_urls:
            print(f"\nTesting: {url}")
            print("-" * 40)
            
            try:
                html, has_shadow = extractor.extract_with_shadow_dom(url, scroll=False)
                
                # Parse with BeautifulSoup to count elements
                soup = BeautifulSoup(html, 'html.parser')
                
                print(f"Shadow DOM detected: {has_shadow}")
                print(f"Total HTML length: {len(html):,} characters")
                print(f"Total links found: {len(soup.find_all('a'))}")
                print(f"Total div elements: {len(soup.find_all('div'))}")
                print(f"Total script tags: {len(soup.find_all('script'))}")
                
            except Exception as e:
                print(f"✗ Error: {str(e)}")


def demo_shadow_host_detection():
    """Demonstrate detection of shadow host elements."""
    print("\n" + "="*60)
    print("DEMO 2: Shadow Host Detection")
    print("="*60)
    
    test_url = "https://shop.polymer-project.org/" # has shadow DOM
    
    with ShadowDOMExtractor(headless=True) as extractor:
        print(f"\nAnalyzing: {test_url}")
        print("-" * 40)
        
        shadow_hosts = extractor.find_shadow_hosts(test_url)
        
        if shadow_hosts:
            print(f"Found {len(shadow_hosts)} shadow host element(s):\n")
            
            # Group by tag name
            tag_counts = {}
            for host in shadow_hosts:
                tag = host['tagName']
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            print("Shadow hosts by tag type:")
            for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"- {tag}: {count}")
                
            print("\nFirst 3 shadow hosts details:")
            for i, host in enumerate(shadow_hosts[:3], 1):
                print(f"\n  [{i}] {host['tagName']}")
                if host['id']:
                    print(f"      ID: {host['id']}")
                if host['className']:
                    print(f"      Classes: {host['className']}")
                print(f"      Mode: {host['shadowRootMode']}")
        else:
            print("No shadow hosts found")


def demo_comparison():
    """Compare extraction with and without shadow DOM handling."""
    print("\n" + "="*60)
    print("DEMO 3: Regular vs Shadow DOM Extraction Comparison")
    print("="*60)
    
    test_url = "https://shop.polymer-project.org/" # has shadow DOM
    
    print(f"\nComparing extraction methods for: {test_url}")
    print("-" * 40)
    
    with ShadowDOMExtractor(headless=True) as extractor:
        # Regular extraction (just page source)
        extractor.setup_driver()
        extractor.driver.get(test_url)
        time.sleep(2)
        regular_html = extractor.driver.page_source
        
        # Shadow DOM extraction
        shadow_html, has_shadow = extractor.extract_with_shadow_dom(test_url)
        
        # Compare
        regular_soup = BeautifulSoup(regular_html, 'html.parser')
        shadow_soup = BeautifulSoup(shadow_html, 'html.parser')
        
        print("\nRegular Extraction (page_source):")
        print(f"  - HTML length: {len(regular_html):,} characters")
        print(f"  - Links found: {len(regular_soup.find_all('a'))}")
        print(f"  - Divs found: {len(regular_soup.find_all('div'))}")
        
        print("\nShadow DOM Extraction:")
        print(f"  - HTML length: {len(shadow_html):,} characters")
        print(f"  - Links found: {len(shadow_soup.find_all('a'))}")
        print(f"  - Divs found: {len(shadow_soup.find_all('div'))}")
        print(f"  - Shadow DOM present: {has_shadow}")
        
        # Calculate difference
        char_diff = len(shadow_html) - len(regular_html)
        link_diff = len(shadow_soup.find_all('a')) - len(regular_soup.find_all('a'))
        div_diff = len(shadow_soup.find_all('div')) - len(regular_soup.find_all('div'))
        
        print("\nDifference (Shadow - Regular):")
        print(f"  - Characters: {char_diff:+,} ({char_diff/len(regular_html)*100:.1f}%)")
        print(f"  - Links: {link_diff:+}")
        print(f"  - Divs: {div_diff:+}")


def demo_custom_website():
    """Allow user to test their own website."""
    print("\n" + "="*60)
    print("DEMO 4: Custom Website Test")
    print("="*60)
    
    print("\nYou can test your own website!")
    print("Enter a URL (or press Enter to skip): ", end="")
    
    custom_url = input().strip()
    
    if not custom_url:
        print("Skipping custom test...")
        return
    
    if not custom_url.startswith(('http://', 'https://')):
        custom_url = 'https://' + custom_url
    
    print(f"\nAnalyzing: {custom_url}")
    print("-" * 40)
    
    with ShadowDOMExtractor(headless=True) as extractor:
        try:
            # Extract with shadow DOM
            html, has_shadow = extractor.extract_with_shadow_dom(custom_url)
            
            # Find shadow hosts
            shadow_hosts = extractor.find_shadow_hosts(custom_url)
            
            # Parse HTML
            soup = BeautifulSoup(html, 'html.parser')
            
            print("\nResults:")
            print(f"Page loaded successfully")
            print(f"Shadow DOM detected: {has_shadow}")
            print(f"Shadow hosts found: {len(shadow_hosts)}")
            print(f"Total HTML length: {len(html):,} characters")
            print(f"Links extracted: {len(soup.find_all('a'))}")
            
            # Show sample of extracted links
            links = soup.find_all('a', href=True)[:5]
            if links:
                print("\nSample of extracted links:")
                for i, link in enumerate(links, 1):
                    text = link.get_text(strip=True)[:50]
                    href = link['href'][:50]
                    print(f"  [{i}] {text or '(no text)'}")
                    print(f"      → {href}...")
                    
        except Exception as e:
            print(f"Error: {str(e)}")


def main():
    print("\n" + "="*60)
    print("SHADOW DOM EXTRACTOR DEMONSTRATION")
    print("="*60)
    
    # Run demos
    demo_basic_extraction()
    demo_shadow_host_detection()
    demo_comparison()
    demo_custom_website()
    
    print("\nDemo completed!")


if __name__ == "__main__":
    main()