# Dataset Directory

Place your Amazon smartwatch reviews dataset here.

## Expected Format

The dataset should be a CSV file named `amazon_reviews.csv` with the following columns:

- `review_text`: The text content of the review
- `rating`: Star rating (1-5)
- `product_name`: Name of the smartwatch product
- `date`: Review date (optional)

## Sample Data Structure

```csv
review_text,rating,product_name,date
"This smartwatch is amazing! Battery lasts for days.",5,Apple Watch Series 8,2024-01-15
"Disappointed with the build quality.",2,Samsung Galaxy Watch,2024-01-14
"Good value for money.",4,Fitbit Versa,2024-01-13
```

## Data Sources

You can obtain smartwatch review data from:

1. **Amazon Product Reviews Dataset** (Kaggle)
   - Search for "Amazon Electronics Reviews" or "Amazon Gadget Reviews"
   - Filter for smartwatch/wearable categories

2. **Web Scraping** (with permission)
   - Use the data_collection.py script
   - Ensure compliance with website terms of service

3. **Public Datasets**
   - UCI Machine Learning Repository
   - Hugging Face Datasets
   - Google Dataset Search

## Preprocessing

The data will be automatically:
- Cleaned and normalized
- Filtered for smartwatch-related products
- Labeled with sentiment (Negative/Neutral/Positive)
- Split into train/validation/test sets
