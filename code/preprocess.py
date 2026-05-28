#!/usr/bin/env python3
"""
TxSpector-SODA Baseline Experiment Preprocessing
================================================

논문 방법론에 따른 전처리:
1. Opcode simplification (PUSH1-32 → PUSH, etc.)
2. N-gram feature extraction (bi-gram, tri-gram)
3. Weight Penalty mechanism (modified TF-IDF)
4. Dataset split (Train/Val/Test)

Author: Claude Code
Date: 2025-11-17
"""

import pandas as pd
import numpy as np
import re
import os
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm
import json


class OpcodePreprocessor:
    """Opcode sequence 전처리 클래스"""

    def __init__(self):
        """초기화"""
        self.opcode_stats = {
            'total': 0,
            'simplified': 0,
            'empty': 0,
            'invalid': 0
        }

    def parse_opcode_sequence(self, seq_str: str) -> List[str]:
        """
        CSV의 opcode_sequence 파싱

        Format: |pc;opcode;value|pc;opcode;value|...
        → Extract only opcodes

        Args:
            seq_str: Raw opcode sequence string

        Returns:
            List of opcode names
        """
        if not seq_str or pd.isna(seq_str):
            self.opcode_stats['empty'] += 1
            return []

        opcodes = []
        try:
            # Split by | and extract opcode (2nd element)
            parts = seq_str.split('|')
            for part in parts:
                if not part.strip():
                    continue

                elements = part.split(';')
                if len(elements) >= 2:
                    opcode = elements[1].strip()
                    if opcode:
                        opcodes.append(opcode)

            self.opcode_stats['total'] += len(opcodes)
            return opcodes

        except Exception as e:
            self.opcode_stats['invalid'] += 1
            print(f"Error parsing opcode sequence: {e}")
            return []

    def simplify_opcode(self, opcode: str) -> str:
        """
        Opcode simplification (논문 Section 4.2)

        Rules:
        - PUSH1-PUSH32 → PUSH
        - DUP1-DUP16 → DUP
        - SWAP1-SWAP16 → SWAP
        - LOG0-LOG4 → LOG

        Args:
            opcode: Original opcode name

        Returns:
            Simplified opcode name
        """
        # PUSH1-PUSH32
        if re.match(r'^PUSH\d+$', opcode):
            self.opcode_stats['simplified'] += 1
            return 'PUSH'

        # DUP1-DUP16
        if re.match(r'^DUP\d+$', opcode):
            self.opcode_stats['simplified'] += 1
            return 'DUP'

        # SWAP1-SWAP16
        if re.match(r'^SWAP\d+$', opcode):
            self.opcode_stats['simplified'] += 1
            return 'SWAP'

        # LOG0-LOG4
        if re.match(r'^LOG\d+$', opcode):
            self.opcode_stats['simplified'] += 1
            return 'LOG'

        return opcode

    def preprocess_sequence(self, seq_str: str) -> str:
        """
        전체 opcode sequence 전처리

        Args:
            seq_str: Raw opcode sequence

        Returns:
            Space-separated simplified opcodes
        """
        # Parse opcodes from CSV format
        opcodes = self.parse_opcode_sequence(seq_str)

        if not opcodes:
            return ""

        # Simplify each opcode
        simplified = [self.simplify_opcode(op) for op in opcodes]

        # Join with space
        return ' '.join(simplified)

    def get_stats(self) -> Dict:
        """전처리 통계 반환"""
        return self.opcode_stats


class DatasetLoader:
    """CSV 데이터 로더"""

    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: CSV 파일이 있는 디렉토리
        """
        self.data_dir = Path(data_dir)
        self.plugin_mapping = {
            'P1': 'S1',  # Re-entrancy
            'P2': 'S2',  # Unexpected Function Invocation
            'P5': 'S3',  # Unchecked Return Value
            'P6': 'S4',  # Missing Transfer Event
            'P7': 'S5',  # Strict Balance Check
            'P8': 'S6',  # Timestamp Dependency
            'P4': 'S7',  # tx.origin (Unknown)
            # P3 not used in paper
        }

    def load_vulnerable_csv(self, plugin: str, max_rows: int = None, max_samples: int = None) -> pd.DataFrame:
        """
        취약점 CSV 로드

        Args:
            plugin: Plugin name (P1-P8)
            max_rows: Maximum rows to load per file (None = all)
            max_samples: Maximum total samples for this plugin (None = all)

        Returns:
            DataFrame with columns: txhash, plugin, block, opcode_sequence
        """
        csv_files = list(self.data_dir.glob(f'slice_*_{plugin}.csv'))

        if not csv_files:
            print(f"Warning: No CSV files found for {plugin}")
            return pd.DataFrame()

        print(f"Loading {plugin} from {len(csv_files)} file(s)...")

        dfs = []
        total_loaded = 0

        for csv_file in csv_files:
            # Stop if we've reached max_samples
            if max_samples is not None and total_loaded >= max_samples:
                break

            try:
                # Calculate how many more rows we need
                rows_to_load = max_rows
                if max_samples is not None:
                    remaining = max_samples - total_loaded
                    if max_rows is not None:
                        rows_to_load = min(max_rows, remaining)
                    else:
                        rows_to_load = remaining

                df = pd.read_csv(
                    csv_file,
                    names=['txhash', 'plugin', 'block', 'opcode_sequence'],
                    skiprows=1,  # Skip header
                    nrows=rows_to_load
                )
                dfs.append(df)
                total_loaded += len(df)
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)

        # Apply final max_samples limit (in case of overshooting)
        if max_samples is not None and len(combined) > max_samples:
            combined = combined.sample(n=max_samples, random_state=42)

        # Map plugin to paper label
        combined['label'] = self.plugin_mapping.get(plugin, plugin)

        print(f"  Loaded {len(combined)} rows for {plugin} (→ {combined['label'].iloc[0]})")

        return combined

    def load_normal_csv(self, max_rows: int = None) -> pd.DataFrame:
        """
        정상 CSV 로드

        Args:
            max_rows: Maximum rows to load

        Returns:
            DataFrame with columns: txhash, label, block, sampling_reason, opcode_sequence
        """
        csv_files = list(self.data_dir.glob('slice_*_normal.csv'))

        if not csv_files:
            print("Warning: No normal CSV files found")
            return pd.DataFrame()

        print(f"Loading normal data from {len(csv_files)} file(s)...")

        dfs = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(
                    csv_file,
                    names=['txhash', 'label', 'block', 'sampling_reason', 'opcode_sequence'],
                    skiprows=1,
                    nrows=max_rows
                )
                # Set label to 'S0' for normal
                df['label'] = 'S0'
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)

        print(f"  Loaded {len(combined)} normal rows")

        return combined

    def load_all_data(
        self,
        max_rows_per_plugin: int = None,
        balanced_sampling: bool = False,
        max_samples_per_plugin: dict = None
    ) -> pd.DataFrame:
        """
        모든 데이터 로드

        Args:
            max_rows_per_plugin: Maximum rows per file per plugin/normal
            balanced_sampling: If True, apply balanced sampling strategy
            max_samples_per_plugin: Dict mapping plugin to max samples (e.g., {'P2': 6000})

        Returns:
            Combined DataFrame
        """
        all_dfs = []

        # Default balanced sampling strategy
        if balanced_sampling and max_samples_per_plugin is None:
            # 희귀 취약점: 모두 사용
            # 빈번 취약점: 6000개로 제한
            max_samples_per_plugin = {
                'P1': None,   # 5,836개 (모두 사용)
                'P2': 6000,   # 1,076,691개 → 6000개
                'P5': None,   # 815개 (모두 사용)
                'P6': 6000,   # 5,270,432개 → 6000개
                'P7': None,   # 1,082개 (모두 사용)
                'P8': 6000,   # 2,123,652개 → 6000개
                'P4': None,   # 13,331개 (Unknown, 모두 사용)
                'normal': 20000  # Normal도 제한
            }
            print("\n[Balanced Sampling Strategy]")
            print("  P1, P5, P7: Use all (rare vulnerabilities)")
            print("  P2, P6, P8: Limit to 6000 (frequent vulnerabilities)")
            print("  P4 (S7): Use all (unknown vulnerability for testing)")
            print("  Normal: Limit to 20000")

        if max_samples_per_plugin is None:
            max_samples_per_plugin = {}

        # Load vulnerable data (S1-S7)
        for plugin in ['P1', 'P2', 'P5', 'P6', 'P7', 'P8', 'P4']:
            max_samples = max_samples_per_plugin.get(plugin, None)
            df = self.load_vulnerable_csv(plugin, max_rows_per_plugin, max_samples)
            if not df.empty:
                # Standardize columns
                df = df[['txhash', 'label', 'block', 'opcode_sequence']]
                all_dfs.append(df)

        # Load normal data (S0)
        max_normal = max_samples_per_plugin.get('normal', None)
        normal_df = self.load_normal_csv(max_rows_per_plugin)
        if not normal_df.empty:
            # Apply max_samples to normal data
            if max_normal is not None and len(normal_df) > max_normal:
                normal_df = normal_df.sample(n=max_normal, random_state=42)
                print(f"  Sampled {max_normal} normal rows (from {len(normal_df)})")
            # Standardize columns
            normal_df = normal_df[['txhash', 'label', 'block', 'opcode_sequence']]
            all_dfs.append(normal_df)

        if not all_dfs:
            raise ValueError("No data loaded!")

        combined = pd.concat(all_dfs, ignore_index=True)

        # Remove duplicates based on txhash
        original_len = len(combined)
        combined = combined.drop_duplicates(subset=['txhash'], keep='first')
        print(f"\nRemoved {original_len - len(combined)} duplicates")

        print(f"\nTotal dataset size: {len(combined)}")
        print("\nLabel distribution:")
        print(combined['label'].value_counts().sort_index())

        return combined


def preprocess_dataset(
    data_dir: str,
    output_dir: str,
    max_rows_per_plugin: int = None,
    test_mode: bool = False,
    balanced_sampling: bool = False,
    max_samples_per_plugin: dict = None
):
    """
    전체 데이터셋 전처리 파이프라인

    Args:
        data_dir: Input CSV directory
        output_dir: Output directory for preprocessed data
        max_rows_per_plugin: Maximum rows to load per plugin (for testing)
        test_mode: If True, run in test mode with small sample
        balanced_sampling: If True, apply balanced sampling strategy
        max_samples_per_plugin: Dict mapping plugin to max samples
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("TxSpector-SODA Dataset Preprocessing")
    print("="*80)

    # Step 1: Load data
    print("\n[Step 1] Loading CSV data...")
    loader = DatasetLoader(data_dir)

    if test_mode:
        print("  Running in TEST MODE - loading 1000 rows per plugin")
        max_rows_per_plugin = 1000

    df = loader.load_all_data(
        max_rows_per_plugin,
        balanced_sampling,
        max_samples_per_plugin
    )

    # Step 2: Preprocess opcodes
    print("\n[Step 2] Preprocessing opcode sequences...")
    preprocessor = OpcodePreprocessor()

    tqdm.pandas(desc="Simplifying opcodes")
    df['opcode_simplified'] = df['opcode_sequence'].progress_apply(
        preprocessor.preprocess_sequence
    )

    # Remove empty sequences
    original_len = len(df)
    df = df[df['opcode_simplified'].str.len() > 0].copy()
    print(f"  Removed {original_len - len(df)} empty sequences")

    # Print preprocessing stats
    stats = preprocessor.get_stats()
    print(f"\nOpcode preprocessing statistics:")
    print(f"  Total opcodes processed: {stats['total']:,}")
    print(f"  Simplified opcodes: {stats['simplified']:,}")
    print(f"  Empty sequences: {stats['empty']:,}")
    print(f"  Invalid sequences: {stats['invalid']:,}")

    # Step 3: Save preprocessed data
    print("\n[Step 3] Saving preprocessed data...")

    # Save full dataset
    output_file = output_path / 'preprocessed_full.csv'
    df.to_csv(output_file, index=False)
    print(f"  Saved full dataset: {output_file}")
    print(f"  Size: {len(df)} rows")

    # Save statistics
    stats_file = output_path / 'preprocessing_stats.json'
    with open(stats_file, 'w') as f:
        json.dump({
            'total_samples': len(df),
            'label_distribution': df['label'].value_counts().to_dict(),
            'opcode_stats': stats,
            'test_mode': test_mode,
            'max_rows_per_plugin': max_rows_per_plugin
        }, f, indent=2)
    print(f"  Saved statistics: {stats_file}")

    # Step 4: Split by label for easy access
    print("\n[Step 4] Splitting by label...")
    for label in sorted(df['label'].unique()):
        label_df = df[df['label'] == label]
        label_file = output_path / f'preprocessed_{label}.csv'
        label_df.to_csv(label_file, index=False)
        print(f"  {label}: {len(label_df)} samples → {label_file.name}")

    print("\n" + "="*80)
    print("Preprocessing completed!")
    print("="*80)

    return df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess TxSpector-SODA dataset')
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/hdd_mnt/data/slicing_checkpoint/extracted_data',
        help='Input CSV directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/home/bjw/change_SODA/SODA/SODA_code/go-ethereum/ml_experiments/data',
        help='Output directory for preprocessed data'
    )
    parser.add_argument(
        '--max-rows',
        type=int,
        default=None,
        help='Maximum rows to load per plugin (for testing)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run in test mode (1000 rows per plugin)'
    )
    parser.add_argument(
        '--balanced',
        action='store_true',
        help='Use balanced sampling (P2,P6,P8 limited to 6000)'
    )

    args = parser.parse_args()

    preprocess_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_rows_per_plugin=args.max_rows,
        test_mode=args.test,
        balanced_sampling=args.balanced
    )
