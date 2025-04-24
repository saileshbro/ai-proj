import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Dataset processing functions - one for each dataset folder

def process_aayamoza_dataset(folder_path="datasets/aayamoza"):
    """
    Process aayamoza dataset and return a standardized DataFrame

    Args:
        folder_path: Path to the aayamoza dataset folder

    Returns:
        DataFrame with standardized columns (text, label, source)
    """
    print(f"Processing aayamoza dataset from {folder_path}")
    all_data = []

    if not os.path.exists(folder_path):
        print(f"Directory {folder_path} not found")
        return pd.DataFrame()

    for file_name in os.listdir(folder_path):
        if not file_name.endswith('.csv'):
            continue

        file_path = os.path.join(folder_path, file_name)
        print(f"  Reading {file_name}")

        try:
            # Read CSV file
            df = pd.read_csv(file_path, quotechar='"', on_bad_lines='warn', engine='python')

            # Clean column names
            df.columns = df.columns.str.strip()

            # Remove unnamed columns
            unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
            if unnamed_cols:
                df = df.drop(columns=unnamed_cols)

            # Check for required columns
            if "Sentences" not in df.columns or "Sentiment" not in df.columns:
                print(f"  Required columns not found in {file_name}. Skipping.")
                continue

            # Extract and rename columns
            df = df[["Sentences", "Sentiment"]].copy()
            df = df.rename(columns={"Sentences": "text", "Sentiment": "label"})

            # Clean data
            df = df.dropna()
            df = df[df['text'].str.strip() != '']

            # Ensure labels are numeric
            df["label"] = pd.to_numeric(df["label"], errors="coerce")
            df = df.dropna(subset=["label"])

            # Map original labels to our new scheme:
            # -1 (negative) -> 0 (negative)
            # 0 (neutral) -> 2 (neutral)
            # 1 (positive) -> 1 (positive)
            df["label"] = df["label"].replace({-1: 0, 0: 2, 1: 1})

            # Add source information
            df["source"] = "aayamoza"

            all_data.append(df)
            print(f"  Successfully processed with {len(df)} rows")

        except Exception as e:
            print(f"  Error processing {file_name}: {str(e)}")

    if not all_data:
        print("  No valid data found in aayamoza dataset.")
        return pd.DataFrame()

    # Combine all files
    result = pd.concat(all_data, ignore_index=True)
    print(f"Total aayamoza data: {len(result)} rows")
    return result


def process_ajhesh7_dataset(folder_path="datasets/ajhesh7"):
    """
    Process ajhesh7 dataset and return a standardized DataFrame

    Args:
        folder_path: Path to the ajhesh7 dataset folder

    Returns:
        DataFrame with standardized columns (text, label, source)
    """
    print(f"Processing ajhesh7 dataset from {folder_path}")
    all_data = []

    if not os.path.exists(folder_path):
        print(f"Directory {folder_path} not found")
        return pd.DataFrame()

    for file_name in os.listdir(folder_path):
        if not file_name.endswith('.csv'):
            continue

        file_path = os.path.join(folder_path, file_name)
        print(f"  Reading {file_name}")

        try:
            # Read CSV file
            df = pd.read_csv(file_path, quotechar='"', on_bad_lines='warn', engine='python')

            # Clean column names
            df.columns = df.columns.str.strip()

            # Remove unnamed columns
            unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
            if unnamed_cols:
                df = df.drop(columns=unnamed_cols)

            # Check for required columns
            if "Sentences" not in df.columns or "Sentiment" not in df.columns:
                print(f"  Required columns not found in {file_name}. Skipping.")
                continue

            # Extract and rename columns
            df = df[["Sentences", "Sentiment"]].copy()
            df = df.rename(columns={"Sentences": "text", "Sentiment": "label"})

            # Clean data
            df = df.dropna()
            df = df[df['text'].str.strip() != '']

            # Ensure labels are numeric
            df["label"] = pd.to_numeric(df["label"], errors="coerce")
            df = df.dropna(subset=["label"])

            # No mapping needed - ajhesh7 already uses 0 (negative), 1 (positive), and 2 (neutral)

            # Add source information
            df["source"] = "ajhesh7"

            all_data.append(df)
            print(f"  Successfully processed with {len(df)} rows")

        except Exception as e:
            print(f"  Error processing {file_name}: {str(e)}")

    if not all_data:
        print("  No valid data found in ajhesh7 dataset.")
        return pd.DataFrame()

    # Combine all files
    result = pd.concat(all_data, ignore_index=True)
    print(f"Total ajhesh7 data: {len(result)} rows")
    return result


def process_shushant_dataset(folder_path="datasets/shushant"):
    """
    Process shushant dataset and return a standardized DataFrame

    Args:
        folder_path: Path to the shushant dataset folder

    Returns:
        DataFrame with standardized columns (text, label, source)
    """
    print(f"Processing shushant dataset from {folder_path}")
    all_data = []

    if not os.path.exists(folder_path):
        print(f"Directory {folder_path} not found")
        return pd.DataFrame()

    for file_name in os.listdir(folder_path):
        if not file_name.endswith('.csv'):
            continue

        file_path = os.path.join(folder_path, file_name)
        print(f"  Reading {file_name}")

        try:
            # Read CSV file
            df = pd.read_csv(file_path, quotechar='"', on_bad_lines='warn', engine='python')

            # Clean column names
            df.columns = df.columns.str.strip()

            # Check for required columns
            if "text" not in df.columns or "label" not in df.columns:
                print(f"  Required columns not found in {file_name}. Skipping.")
                continue

            # Extract columns (already named correctly)
            df = df[["text", "label"]].copy()

            # Clean data
            df = df.dropna()
            df = df[df['text'].str.strip() != '']

            # Ensure labels are numeric
            df["label"] = pd.to_numeric(df["label"], errors="coerce")
            df = df.dropna(subset=["label"])

            # The dataset already uses the desired mapping:
            # 0 (negative) -> 0 (negative)
            # 1 (positive) -> 1 (positive)
            # 2 (neutral) -> 2 (neutral)
            # No mapping needed

            # Add source information
            df["source"] = "shushant"

            all_data.append(df)
            print(f"  Successfully processed with {len(df)} rows")

        except Exception as e:
            print(f"  Error processing {file_name}: {str(e)}")

    if not all_data:
        print("  No valid data found in shushant dataset.")
        return pd.DataFrame()

    # Combine all files
    result = pd.concat(all_data, ignore_index=True)
    print(f"Total shushant data: {len(result)} rows")
    return result


def process_smahesh_dataset(folder_path="datasets/smahesh"):
    """
    Process smahesh dataset and return a standardized DataFrame

    Args:
        folder_path: Path to the smahesh dataset folder

    Returns:
        DataFrame with standardized columns (text, label, source)
    """
    print(f"Processing smahesh dataset from {folder_path}")
    all_data = []

    if not os.path.exists(folder_path):
        print(f"Directory {folder_path} not found")
        return pd.DataFrame()

    for file_name in os.listdir(folder_path):
        if not file_name.endswith('.csv'):
            continue

        file_path = os.path.join(folder_path, file_name)
        print(f"  Reading {file_name}")

        try:
            # First read file as text to fix the CSV formatting issues
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Fix common CSV issues with quote handling
            fixed_lines = []
            for line in lines:
                # Skip empty lines
                if line.strip() == '':
                    continue

                # Fix quotes by ensuring they're properly paired
                if line.count('"') % 2 != 0:
                    # Add closing quote if missing
                    line = line.rstrip('\n') + '"\n'

                fixed_lines.append(line)

            # Write fixed content to a temporary file
            temp_file_path = file_path + '.temp.csv'
            with open(temp_file_path, 'w', encoding='utf-8') as f:
                f.writelines(fixed_lines)

            # Read the fixed CSV file
            df = pd.read_csv(temp_file_path, quotechar='"', on_bad_lines='skip', engine='python')

            # Clean up temporary file
            os.remove(temp_file_path)

            # Clean column names
            df.columns = df.columns.str.strip()

            # Check for required columns
            if "Data" not in df.columns or "Label" not in df.columns:
                print(f"  Required columns not found in {file_name}. Skipping.")
                continue

            # Extract and rename columns
            df = df[["Data", "Label"]].copy()
            df = df.rename(columns={"Data": "text", "Label": "label"})

            # Clean data
            df = df.dropna()
            df = df[df['text'].str.strip() != '']

            # Ensure labels are numeric
            df["label"] = pd.to_numeric(df["label"], errors="coerce")
            df = df.dropna(subset=["label"])

            # Map the smahesh labels to our new scheme:
            # 0 (negative) - not present in the dataset
            # 1 (positive) -> 1 (positive)
            # 2 (neutral) -> 2 (neutral)
            # No mapping needed as the scheme already aligns with our desired output

            # Add source information
            df["source"] = "smahesh"

            all_data.append(df)
            print(f"  Successfully processed with {len(df)} rows")

        except Exception as e:
            print(f"  Error processing {file_name}: {str(e)}")

    if not all_data:
        print("  No valid data found in smahesh dataset.")
        return pd.DataFrame()

    # Combine all files
    result = pd.concat(all_data, ignore_index=True)
    print(f"Total smahesh data: {len(result)} rows")
    return result


# Main functions for dataset preparation and splitting

def prepare_dataset():
    """
    Prepare and combine datasets from all sources

    Returns:
        DataFrame with standardized columns (text, label, source)
    """
    # Process each dataset
    df_aayamoza = process_aayamoza_dataset()
    df_ajhesh7 = process_ajhesh7_dataset()
    df_shushant = process_shushant_dataset()
    df_smahesh = process_smahesh_dataset()

    # Collect all non-empty datasets
    datasets = []
    if not df_aayamoza.empty:
        datasets.append(df_aayamoza)
    if not df_ajhesh7.empty:
        datasets.append(df_ajhesh7)
    if not df_shushant.empty:
        datasets.append(df_shushant)
    if not df_smahesh.empty:
        datasets.append(df_smahesh)

    if not datasets:
        raise ValueError("No data was successfully loaded. Check your dataset paths and formats.")

    # Combine all datasets
    df = pd.concat(datasets, ignore_index=True)

    # Final cleanup
    # Remove duplicates
    df = df.drop_duplicates(subset=["text"])

    # Ensure label values are correct (0, 1, 2)
    valid_labels = [0, 1, 2]  # 0 (negative), 1 (positive), 2 (neutral)
    invalid_labels = [label for label in df["label"].unique() if label not in valid_labels]
    if invalid_labels:
        print(f"Warning: Found invalid labels {invalid_labels}. These will be removed.")
        df = df[df["label"].isin(valid_labels)]

    # Check for missing values
    missing_values = df.isna().sum()
    if missing_values.sum() > 0:
        print(f"Found missing values: {missing_values}")
        df = df.dropna()

    # Convert label to int
    df["label"] = df["label"].astype(int)

    # Print dataset statistics
    print("\nFinal dataset shape:", df.shape)
    print("\nLabel distribution:")
    print(df["label"].value_counts())
    print("\nSource distribution:")
    print(df["source"].value_counts())

    return df


def split_dataset(df, train_ratio=0.8, random_state=42):
    """
    Split dataset into training and testing sets

    Args:
        df: DataFrame with text, label columns
        train_ratio: Proportion of data for training (default: 0.8)
        random_state: Random seed for reproducibility

    Returns:
        train_df, test_df: Split DataFrames
    """
    # Create stratified train/test split
    train_df, test_df = train_test_split(
        df,
        train_size=train_ratio,
        random_state=random_state,
        stratify=df["label"]
    )

    print(f"\nTraining set: {train_df.shape[0]} samples")
    print(f"Testing set: {test_df.shape[0]} samples")

    print("\nTraining set label distribution:")
    print(train_df["label"].value_counts())
    print("\nTesting set label distribution:")
    print(test_df["label"].value_counts())

    return train_df, test_df


def save_datasets(train_df, test_df, output_dir="."):
    """
    Save training and testing datasets to CSV files

    Args:
        train_df: Training DataFrame
        test_df: Testing DataFrame
        output_dir: Directory to save files (default: current directory)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save only text and label columns
    train_df[["text", "label"]].to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_df[["text", "label"]].to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print(f"\nDatasets saved to {output_dir}")


# Command-line interface
if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser(description="Prepare dataset for Nepali sentiment analysis")
    # parser.add_argument('--output_dir', help='Directory to save output files', default='.')
    # parser.add_argument('--train_ratio', type=float, help='Ratio of training data (0-1)', default=0.8)

    # args = parser.parse_args()

    # Process datasets
    df = prepare_dataset()

    # Display sample data
    print("\nSample data:")
    print(df[["text", "label"]].head())

    # Split dataset
    train_df, test_df = split_dataset(df, 0.8)

    # Save datasets
    save_datasets(train_df, test_df, 'processed')