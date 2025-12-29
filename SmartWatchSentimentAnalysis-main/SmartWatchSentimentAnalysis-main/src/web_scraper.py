"""
Web scraper for collecting real smartwatch reviews from online sources
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from typing import List, Dict
import json
from datetime import datetime


class AmazonReviewScraper:
    """
    Scraper for Amazon smartwatch reviews
    Note: For educational purposes. Always respect robots.txt and terms of service.
    """
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        self.base_url = "https://www.amazon.com"
    
    def scrape_product_reviews(self, product_asin: str, max_pages: int = 5) -> List[Dict]:
        """
        Scrape reviews for a specific product
        
        Args:
            product_asin: Amazon product ASIN
            max_pages: Maximum number of review pages to scrape
        
        Returns:
            List of review dictionaries
        """
        reviews = []
        
        for page in range(1, max_pages + 1):
            url = f"{self.base_url}/product-reviews/{product_asin}/ref=cm_cr_arp_d_paging_btm_next_{page}?pageNumber={page}"
            
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                review_divs = soup.find_all('div', {'data-hook': 'review'})
                
                for review_div in review_divs:
                    try:
                        review_data = self._extract_review_data(review_div)
                        if review_data:
                            reviews.append(review_data)
                    except Exception as e:
                        print(f"Error extracting review: {e}")
                        continue
                
                # Be respectful - add delay between requests
                time.sleep(2)
                
            except Exception as e:
                print(f"Error scraping page {page}: {e}")
                break
        
        return reviews
    
    def _extract_review_data(self, review_div) -> Dict:
        """Extract review data from BeautifulSoup element"""
        try:
            # Extract rating
            rating_elem = review_div.find('i', {'data-hook': 'review-star-rating'})
            rating = float(rating_elem.text.split()[0]) if rating_elem else None
            
            # Extract review text
            text_elem = review_div.find('span', {'data-hook': 'review-body'})
            review_text = text_elem.text.strip() if text_elem else ""
            
            # Extract title
            title_elem = review_div.find('a', {'data-hook': 'review-title'})
            title = title_elem.text.strip() if title_elem else ""
            
            # Extract date
            date_elem = review_div.find('span', {'data-hook': 'review-date'})
            date = date_elem.text.strip() if date_elem else ""
            
            # Extract verified purchase
            verified_elem = review_div.find('span', {'data-hook': 'avp-badge'})
            verified = verified_elem is not None
            
            return {
                'rating': rating,
                'title': title,
                'review_text': review_text,
                'date': date,
                'verified_purchase': verified,
                'scraped_at': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Error in _extract_review_data: {e}")
            return None


class ReviewDataCollector:
    """
    Unified interface for collecting reviews from multiple sources
    """
    
    def __init__(self):
        self.amazon_scraper = AmazonReviewScraper()
    
    def collect_smartwatch_reviews(self, source: str = 'amazon', max_reviews: int = 500) -> pd.DataFrame:
        """
        Collect smartwatch reviews from specified source
        
        Args:
            source: Data source ('amazon', 'dataset', 'api')
            max_reviews: Maximum number of reviews to collect
        
        Returns:
            DataFrame with reviews
        """
        if source == 'amazon':
            return self._collect_from_amazon(max_reviews)
        elif source == 'dataset':
            return self._load_from_dataset()
        elif source == 'api':
            return self._collect_from_api(max_reviews)
        else:
            raise ValueError(f"Unknown source: {source}")
    
    def _collect_from_amazon(self, max_reviews: int) -> pd.DataFrame:
        """
        Collect reviews from Amazon
        Note: This is a template. Actual implementation requires proper setup.
        """
        # Popular smartwatch ASINs (examples)
        smartwatch_asins = [
            'B0BDKBGZ8V',  # Apple Watch Series 8
            'B0B2MLWQF9',  # Samsung Galaxy Watch 5
            'B0B4MWCFV4',  # Fitbit Versa 4
            'B0B3QLXQVZ',  # Garmin Venu 2
        ]
        
        all_reviews = []
        reviews_per_product = max_reviews // len(smartwatch_asins)
        
        for asin in smartwatch_asins:
            print(f"Collecting reviews for ASIN: {asin}")
            reviews = self.amazon_scraper.scrape_product_reviews(
                asin, 
                max_pages=reviews_per_product // 10
            )
            all_reviews.extend(reviews)
            
            if len(all_reviews) >= max_reviews:
                break
        
        df = pd.DataFrame(all_reviews)
        return df
    
    def _load_from_dataset(self) -> pd.DataFrame:
        """Load from existing dataset"""
        try:
            df = pd.read_csv('data/amazon_reviews.csv')
            return df
        except FileNotFoundError:
            print("Dataset not found. Using sample data.")
            return self._generate_sample_data()
    
    def _collect_from_api(self, max_reviews: int) -> pd.DataFrame:
        """
        Collect from API (e.g., Hugging Face datasets)
        """
        try:
            from datasets import load_dataset
            
            # Load Amazon reviews dataset from Hugging Face
            dataset = load_dataset('amazon_us_reviews', 'Wireless_v1_00', split='train')
            
            # Filter for smartwatch-related products
            df = dataset.to_pandas()
            smartwatch_keywords = ['smartwatch', 'smart watch', 'fitness tracker', 
                                   'apple watch', 'galaxy watch', 'fitbit', 'garmin']
            
            mask = df['product_title'].str.lower().str.contains('|'.join(smartwatch_keywords), na=False)
            df = df[mask].head(max_reviews)
            
            # Rename columns to match our schema
            df = df.rename(columns={
                'review_body': 'review_text',
                'star_rating': 'rating',
                'review_date': 'date'
            })
            
            return df[['review_text', 'rating', 'product_title', 'date']]
            
        except Exception as e:
            print(f"Error loading from API: {e}")
            return self._load_from_dataset()
    
    def _generate_sample_data(self) -> pd.DataFrame:
        """Generate expanded sample data for demonstration"""
        return pd.read_csv('data/amazon_reviews.csv')
    
    def save_reviews(self, df: pd.DataFrame, filename: str = 'data/collected_reviews.csv'):
        """Save collected reviews to file"""
        df.to_csv(filename, index=False)
        print(f"Saved {len(df)} reviews to {filename}")


def collect_and_prepare_data(source: str = 'dataset', max_reviews: int = 1000):
    """
    Main function to collect and prepare review data
    
    Args:
        source: Data source ('amazon', 'dataset', 'api')
        max_reviews: Maximum number of reviews to collect
    
    Returns:
        DataFrame with collected and prepared reviews
    """
    collector = ReviewDataCollector()
    
    print(f"Collecting reviews from {source}...")
    df = collector.collect_smartwatch_reviews(source, max_reviews)
    
    print(f"Collected {len(df)} reviews")
    
    # Save collected data
    collector.save_reviews(df, 'data/collected_reviews.csv')
    
    return df


if __name__ == "__main__":
    # Example usage
    print("="*80)
    print("SMARTWATCH REVIEW DATA COLLECTION")
    print("="*80)
    
    # Collect from dataset (safest option)
    df = collect_and_prepare_data(source='dataset', max_reviews=1000)
    
    print(f"\nCollected {len(df)} reviews")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nSample review:")
    print(df.iloc[0]['review_text'][:200] + "...")
