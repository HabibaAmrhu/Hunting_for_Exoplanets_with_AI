"""
Label harmonization utilities for multi-survey exoplanet data.
Standardizes different labeling schemes across Kepler, TESS, and K2.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from enum import Enum


class DispositionType(Enum):
    """Standard disposition types for exoplanet candidates."""
    CONFIRMED = "CONFIRMED"
    CANDIDATE = "CANDIDATE"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    UNKNOWN = "UNKNOWN"


class LabelHarmonizer:
    """
    Harmonizes labels across different exoplanet surveys.
    """
    
    def __init__(self):
        """Initialize the label harmonizer."""
        self.logger = logging.getLogger(__name__)
        
        # Kepler/KOI label mappings
        self.koi_mappings = {
            'CONFIRMED': DispositionType.CONFIRMED,
            'CANDIDATE': DispositionType.CANDIDATE,
            'FALSE POSITIVE': DispositionType.FALSE_POSITIVE,
            'NOT DISPOSITIONED': DispositionType.UNKNOWN
        }
        
        # TESS TFOPWG label mappings
        self.tess_mappings = {
            'CP': DispositionType.CONFIRMED,      # Confirmed Planet
            'KP': DispositionType.CONFIRMED,      # Known Planet
            'PC': DispositionType.CANDIDATE,      # Planet Candidate
            'APC': DispositionType.CANDIDATE,     # Ambiguous Planet Candidate
            'FP': DispositionType.FALSE_POSITIVE, # False Positive
            'FA': DispositionType.FALSE_POSITIVE, # False Alarm
            'IS': DispositionType.UNKNOWN,        # Insufficient Data
            'O': DispositionType.UNKNOWN          # Other
        }
        
        # K2 label mappings
        self.k2_mappings = {
            'CONFIRMED': DispositionType.CONFIRMED,
            'CANDIDATE': DispositionType.CANDIDATE,
            'FALSE POSITIVE': DispositionType.FALSE_POSITIVE
        }
        
        # Binary label mappings for ML
        self.binary_mappings = {
            DispositionType.CONFIRMED: 1,
            DispositionType.CANDIDATE: 1,  # Treat candidates as positive
            DispositionType.FALSE_POSITIVE: 0,
            DispositionType.UNKNOWN: -1  # Exclude from training
        }
    
    def harmonize_koi_labels(self, df: pd.DataFrame, disposition_col: str = 'koi_disposition') -> pd.DataFrame:
        """
        Harmonize KOI/Kepler disposition labels.
        
        Args:
            df: DataFrame with KOI data
            disposition_col: Column name containing dispositions
            
        Returns:
            DataFrame with harmonized labels
        """
        df = df.copy()
        
        # Clean and standardize disposition strings
        df[disposition_col] = df[disposition_col].astype(str).str.upper().str.strip()
        
        # Map to standard dispositions
        df['standard_disposition'] = df[disposition_col].map(self.koi_mappings)
        df['standard_disposition'] = df['standard_disposition'].fillna(DispositionType.UNKNOWN)
        
        # Create binary labels
        df['binary_label'] = df['standard_disposition'].map(self.binary_mappings)
        
        # Log statistics
        disposition_counts = df['standard_disposition'].value_counts()
        self.logger.info(f"KOI label distribution: {disposition_counts.to_dict()}")
        
        return df
    
    def harmonize_tess_labels(self, df: pd.DataFrame, disposition_col: str = 'tfopwg_disp') -> pd.DataFrame:
        """
        Harmonize TESS TFOPWG disposition labels.
        
        Args:
            df: DataFrame with TESS data
            disposition_col: Column name containing dispositions
            
        Returns:
            DataFrame with harmonized labels
        """
        df = df.copy()
        
        # Clean and standardize disposition strings
        df[disposition_col] = df[disposition_col].astype(str).str.upper().str.strip()
        
        # Map to standard dispositions
        df['standard_disposition'] = df[disposition_col].map(self.tess_mappings)
        df['standard_disposition'] = df['standard_disposition'].fillna(DispositionType.UNKNOWN)
        
        # Create binary labels
        df['binary_label'] = df['standard_disposition'].map(self.binary_mappings)
        
        # Log statistics
        disposition_counts = df['standard_disposition'].value_counts()
        self.logger.info(f"TESS label distribution: {disposition_counts.to_dict()}")
        
        return df
    
    def harmonize_k2_labels(self, df: pd.DataFrame, disposition_col: str = 'disposition') -> pd.DataFrame:
        """
        Harmonize K2 disposition labels.
        
        Args:
            df: DataFrame with K2 data
            disposition_col: Column name containing dispositions
            
        Returns:
            DataFrame with harmonized labels
        """
        df = df.copy()
        
        # Clean and standardize disposition strings
        df[disposition_col] = df[disposition_col].astype(str).str.upper().str.strip()
        
        # Map to standard dispositions
        df['standard_disposition'] = df[disposition_col].map(self.k2_mappings)
        df['standard_disposition'] = df['standard_disposition'].fillna(DispositionType.UNKNOWN)
        
        # Create binary labels
        df['binary_label'] = df['standard_disposition'].map(self.binary_mappings)
        
        # Log statistics
        disposition_counts = df['standard_disposition'].value_counts()
        self.logger.info(f"K2 label distribution: {disposition_counts.to_dict()}")
        
        return df
    
    def create_balanced_dataset(
        self, 
        df: pd.DataFrame, 
        balance_method: str = 'undersample',
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Create a balanced dataset for training.
        
        Args:
            df: DataFrame with harmonized labels
            balance_method: Method for balancing ('undersample', 'oversample', 'none')
            random_state: Random seed for reproducibility
            
        Returns:
            Balanced DataFrame
        """
        # Filter out unknown labels
        valid_df = df[df['binary_label'] != -1].copy()
        
        if balance_method == 'none':
            return valid_df
        
        # Get class distributions
        positive_samples = valid_df[valid_df['binary_label'] == 1]
        negative_samples = valid_df[valid_df['binary_label'] == 0]
        
        pos_count = len(positive_samples)
        neg_count = len(negative_samples)
        
        self.logger.info(f"Original distribution - Positive: {pos_count}, Negative: {neg_count}")
        
        if balance_method == 'undersample':
            # Undersample majority class
            min_count = min(pos_count, neg_count)
            
            balanced_positive = positive_samples.sample(n=min_count, random_state=random_state)
            balanced_negative = negative_samples.sample(n=min_count, random_state=random_state)
            
            balanced_df = pd.concat([balanced_positive, balanced_negative])
            
        elif balance_method == 'oversample':
            # Oversample minority class
            max_count = max(pos_count, neg_count)
            
            if pos_count < neg_count:
                # Oversample positive class
                oversampled_positive = positive_samples.sample(
                    n=max_count, replace=True, random_state=random_state
                )
                balanced_df = pd.concat([oversampled_positive, negative_samples])
            else:
                # Oversample negative class
                oversampled_negative = negative_samples.sample(
                    n=max_count, replace=True, random_state=random_state
                )
                balanced_df = pd.concat([positive_samples, oversampled_negative])
        
        else:
            raise ValueError(f"Unknown balance method: {balance_method}")
        
        # Shuffle the balanced dataset
        balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        final_pos = sum(balanced_df['binary_label'] == 1)
        final_neg = sum(balanced_df['binary_label'] == 0)
        
        self.logger.info(f"Balanced distribution - Positive: {final_pos}, Negative: {final_neg}")
        
        return balanced_df
    
    def get_label_statistics(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Get comprehensive label statistics.
        
        Args:
            df: DataFrame with harmonized labels
            
        Returns:
            Dictionary of statistics
        """
        stats = {}
        
        # Overall statistics
        total_samples = len(df)
        valid_samples = len(df[df['binary_label'] != -1])
        positive_samples = sum(df['binary_label'] == 1)
        negative_samples = sum(df['binary_label'] == 0)
        unknown_samples = sum(df['binary_label'] == -1)
        
        stats['overall'] = {
            'total_samples': total_samples,
            'valid_samples': valid_samples,
            'positive_samples': positive_samples,
            'negative_samples': negative_samples,
            'unknown_samples': unknown_samples,
            'positive_rate': positive_samples / valid_samples if valid_samples > 0 else 0,
            'class_balance_ratio': positive_samples / negative_samples if negative_samples > 0 else float('inf')
        }
        
        # Survey-specific statistics
        if 'survey' in df.columns:
            stats['by_survey'] = {}
            for survey in df['survey'].unique():
                survey_df = df[df['survey'] == survey]
                survey_valid = len(survey_df[survey_df['binary_label'] != -1])
                survey_pos = sum(survey_df['binary_label'] == 1)
                survey_neg = sum(survey_df['binary_label'] == 0)
                
                stats['by_survey'][survey] = {
                    'total_samples': len(survey_df),
                    'valid_samples': survey_valid,
                    'positive_samples': survey_pos,
                    'negative_samples': survey_neg,
                    'positive_rate': survey_pos / survey_valid if survey_valid > 0 else 0
                }
        
        # Disposition distribution
        if 'standard_disposition' in df.columns:
            disposition_counts = df['standard_disposition'].value_counts()
            stats['disposition_distribution'] = disposition_counts.to_dict()
        
        return stats
    
    def export_label_mapping_report(self, output_file: str) -> str:
        """
        Export a comprehensive label mapping report.
        
        Args:
            output_file: Path to save the report
            
        Returns:
            Report content as string
        """
        report_lines = [
            "# Exoplanet Label Harmonization Report",
            "",
            "## Label Mapping Schemes",
            "",
            "### Kepler/KOI Mappings",
        ]
        
        for original, standard in self.koi_mappings.items():
            binary_label = self.binary_mappings[standard]
            report_lines.append(f"- `{original}` → `{standard.value}` → Binary: {binary_label}")
        
        report_lines.extend([
            "",
            "### TESS TFOPWG Mappings",
        ])
        
        for original, standard in self.tess_mappings.items():
            binary_label = self.binary_mappings[standard]
            report_lines.append(f"- `{original}` → `{standard.value}` → Binary: {binary_label}")
        
        report_lines.extend([
            "",
            "### K2 Mappings",
        ])
        
        for original, standard in self.k2_mappings.items():
            binary_label = self.binary_mappings[standard]
            report_lines.append(f"- `{original}` → `{standard.value}` → Binary: {binary_label}")
        
        report_lines.extend([
            "",
            "## Binary Label Strategy",
            "",
            "- **Positive (1)**: CONFIRMED and CANDIDATE dispositions",
            "- **Negative (0)**: FALSE_POSITIVE dispositions", 
            "- **Excluded (-1)**: UNKNOWN dispositions (not used in training)",
            "",
            "## Usage",
            "",
            "```python",
            "from src.data.label_harmonizer import LabelHarmonizer",
            "",
            "harmonizer = LabelHarmonizer()",
            "koi_df = harmonizer.harmonize_koi_labels(koi_df)",
            "tess_df = harmonizer.harmonize_tess_labels(tess_df)",
            "balanced_df = harmonizer.create_balanced_dataset(combined_df)",
            "```"
        ])
        
        report_content = "\n".join(report_lines)
        
        # Save report if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_content)
        
        return report_content


# Convenience functions
def harmonize_all_labels(
    koi_df: Optional[pd.DataFrame] = None,
    tess_df: Optional[pd.DataFrame] = None,
    k2_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Harmonize labels from all provided datasets and combine them.
    
    Args:
        koi_df: KOI DataFrame
        tess_df: TESS DataFrame
        k2_df: K2 DataFrame
        
    Returns:
        Combined DataFrame with harmonized labels
    """
    harmonizer = LabelHarmonizer()
    combined_dfs = []
    
    if koi_df is not None:
        koi_harmonized = harmonizer.harmonize_koi_labels(koi_df)
        koi_harmonized['survey'] = 'Kepler'
        combined_dfs.append(koi_harmonized)
    
    if tess_df is not None:
        tess_harmonized = harmonizer.harmonize_tess_labels(tess_df)
        tess_harmonized['survey'] = 'TESS'
        combined_dfs.append(tess_harmonized)
    
    if k2_df is not None:
        k2_harmonized = harmonizer.harmonize_k2_labels(k2_df)
        k2_harmonized['survey'] = 'K2'
        combined_dfs.append(k2_harmonized)
    
    if not combined_dfs:
        raise ValueError("At least one dataset must be provided")
    
    combined_df = pd.concat(combined_dfs, ignore_index=True)
    return combined_df


def create_competition_ready_dataset(
    metadata_files: Dict[str, str],
    output_dir: str,
    balance_method: str = 'undersample'
) -> Dict[str, str]:
    """
    Create competition-ready datasets from metadata files.
    
    Args:
        metadata_files: Dictionary mapping survey names to file paths
        output_dir: Output directory for processed datasets
        balance_method: Method for class balancing
        
    Returns:
        Dictionary of created file paths
    """
    import os
    
    harmonizer = LabelHarmonizer()
    
    # Load datasets
    datasets = {}
    if 'koi' in metadata_files:
        datasets['koi'] = pd.read_csv(metadata_files['koi'], comment='#')
    if 'tess' in metadata_files:
        datasets['tess'] = pd.read_csv(metadata_files['tess'], comment='#')
    if 'k2' in metadata_files:
        datasets['k2'] = pd.read_csv(metadata_files['k2'], comment='#')
    
    # Harmonize labels
    combined_df = harmonize_all_labels(**datasets)
    
    # Create balanced dataset
    balanced_df = harmonizer.create_balanced_dataset(combined_df, balance_method=balance_method)
    
    # Create train/val/test splits
    train_size = int(0.7 * len(balanced_df))
    val_size = int(0.15 * len(balanced_df))
    
    train_df = balanced_df[:train_size]
    val_df = balanced_df[train_size:train_size + val_size]
    test_df = balanced_df[train_size + val_size:]
    
    # Save datasets
    os.makedirs(output_dir, exist_ok=True)
    
    output_files = {}
    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        file_path = os.path.join(output_dir, f'{split_name}_harmonized.csv')
        split_df.to_csv(file_path, index=False)
        output_files[split_name] = file_path
    
    # Save full datasets
    combined_file = os.path.join(output_dir, 'combined_harmonized.csv')
    combined_df.to_csv(combined_file, index=False)
    output_files['combined'] = combined_file
    
    balanced_file = os.path.join(output_dir, 'balanced_harmonized.csv')
    balanced_df.to_csv(balanced_file, index=False)
    output_files['balanced'] = balanced_file
    
    # Generate statistics report
    stats = harmonizer.get_label_statistics(balanced_df)
    report_file = os.path.join(output_dir, 'label_harmonization_report.md')
    harmonizer.export_label_mapping_report(report_file)
    output_files['report'] = report_file
    
    return output_files