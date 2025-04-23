# Dataset Guide and Data Preprocessing 📊

This guide explains everything about the dataset used in the Nepali Sentiment Analysis project, including its structure, preprocessing steps, and important considerations.

## Dataset Overview 📝

### Structure
The project uses two CSV files:
```bash
train.csv  # Training dataset (1,995 samples)
test.csv   # Test dataset (5,999 samples)
```

### Data Format
Each dataset contains two columns:
```csv
text,label
"तपाईंको फिल्म राम्रो छ",1
"यो फिल्म मन परेन",0
"ठिकै छ",2
```

### Label Distribution
- **0**: Negative sentiment
- **1**: Positive sentiment
- **2**: Neutral sentiment

## Data Preprocessing Steps 🔄

### 1. Loading Data
```python
import pandas as pd

# Load datasets
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
```

### 2. Handling Missing Values
```python
# Check for missing values
train_na = df_train.isna().sum()
test_na = df_test.isna().sum()

# Remove rows with missing values
df_train = df_train.dropna()
df_test = df_test.dropna()

# Reset indices after dropping
df_train = df_train.reset_index(drop=True)
```

### 3. Label Processing
```python
# Define valid labels
valid_labels = ['0', '1', '2']

# Filter and convert labels to integers
df_train = df_train.loc[df_train['label'].isin(valid_labels)]
df_train.loc[:, 'label'] = df_train['label'].astype(int)
```

## Text Processing with BERT 🔤

### 1. Tokenization
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

# Example tokenization
text = "तपाईंको फिल्म राम्रो छ"
tokens = tokenizer.tokenize(text)
print(tokens)  # ['तपाई', '##ं', '##को', 'फिल्म', 'राम्रो', 'छ']
```

### 2. Encoding
```python
# Convert text to BERT input format
train_tokens = tokenizer.batch_encode_plus(
    df_train['text'].tolist(),
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors='pt'
)
```

## Data Analysis 📈

### 1. Dataset Size
```python
print(f"Training samples: {len(df_train)}")
print(f"Testing samples: {len(df_test)}")
```

### 2. Label Distribution
```python
import matplotlib.pyplot as plt

# Plot label distribution
plt.figure(figsize=(10, 5))
df_train['label'].value_counts().plot(kind='bar')
plt.title('Label Distribution in Training Data')
plt.xlabel('Sentiment')
plt.ylabel('Count')
```

### 3. Text Length Analysis
```python
# Analyze text lengths
text_lengths = df_train['text'].str.len()
print(f"Average text length: {text_lengths.mean():.2f}")
print(f"Max text length: {text_lengths.max()}")
```

## Data Quality Checks ✅

### 1. Label Validation
```python
def validate_labels(df):
    invalid_labels = df[~df['label'].isin([0, 1, 2])]
    return len(invalid_labels) == 0
```

### 2. Text Validation
```python
def validate_text(df):
    # Check for empty strings
    empty_texts = df[df['text'].str.strip() == '']
    return len(empty_texts) == 0
```

### 3. Character Set Validation
```python
def validate_nepali_text(text):
    # Nepali Unicode range
    nepali_pattern = re.compile(r'[\u0900-\u097F]+')
    return bool(nepali_pattern.search(text))
```

## Data Augmentation Techniques (Optional) 🔄

### 1. Back Translation
```python
from transformers import MarianMTModel

def back_translate(text):
    # Nepali → English → Nepali
    # This helps create paraphrased versions
    pass
```

### 2. Synonym Replacement
```python
def replace_synonyms(text):
    # Replace words with their synonyms
    # Requires Nepali WordNet or similar resource
    pass
```

## Best Practices 👍

### 1. Data Splitting
- Maintain class distribution
- Random stratified split
- Proper validation set

### 2. Text Preprocessing
- Keep original case (BERT is case-sensitive)
- Preserve special characters
- Handle numbers consistently

### 3. Memory Management
- Process data in batches
- Use generators for large datasets
- Clear memory when possible

## Common Issues and Solutions 🔧

### 1. Imbalanced Classes
```python
from sklearn.utils.class_weight import compute_class_weight

# Calculate class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(df_train['label']),
    y=df_train['label']
)
```

### 2. Long Sequences
```python
# Handle texts longer than max_length
def truncate_long_texts(texts, max_length=512):
    return [text[:max_length] for text in texts]
```

### 3. Special Characters
```python
def clean_special_chars(text):
    # Keep necessary punctuation
    # Remove unwanted characters
    pass
```

## Resources 📚

### Nepali Language Resources
1. [Nepali WordNet](http://www.cfilt.iitb.ac.in/indowordnet/)
2. [Nepali NLP Tools](https://github.com/sushil79g/Nepali_nlp)
3. [Nepali Unicode Guide](https://unicode.org/charts/PDF/U0900.pdf)

### Data Processing Tools
1. [pandas Documentation](https://pandas.pydata.org/docs/)
2. [BERT Tokenizer Guide](https://huggingface.co/docs/transformers/main_classes/tokenizer)
3. [Data Cleaning Best Practices](https://scikit-learn.org/stable/modules/preprocessing.html)

## Next Steps 🎯

1. Proceed to [Model Training](05-MODEL_TRAINING.md)
2. Review [Technical Details](03-TECHNICAL_DETAILS.md)
3. Check [Usage Guide](06-USAGE_GUIDE.md)