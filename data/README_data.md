# Dataset Description and Reproduction Instructions

## Overview

The dataset used in this study consists of **33,202 Ethereum mainnet transaction samples**, each labeled with one of 9 classes (S0: normal, S1–S8: vulnerability types).

Each sample corresponds to a single Ethereum transaction and contains the full EVM opcode execution trace collected via transaction replay.

---

## Provided File

### `dataset_txhash_label.csv`

| Column | Description |
|--------|-------------|
| `txhash` | Ethereum transaction hash (unique identifier) |
| `label` | Vulnerability class label: S0 (normal), S1–S8 (vulnerability types) |
| `block` | Ethereum block number in which the transaction was included |

This file (2.5 MB, 33,202 rows) enables reproduction of the full dataset without redistributing raw opcode sequences (12 GB).

---

## Vulnerability Label Definitions

| Label | Vulnerability Type | SODA Plugin | Samples |
|-------|--------------------|-------------|---------|
| S0 | Normal (Benign) | — | 5,624 |
| S1 | Re-entrancy (RE) | P1 | 1,949 |
| S2 | Unexpected Function Invocation (UFI) | P2 | 6,000 |
| S3 | No Check After Invocation (NCAI) | P5 | 748 |
| S4 | Missing Transfer Event (MTE) | P6 | 5,997 |
| S5 | Strict Check for Balance (SCB) | P7 | 710 |
| S6 | Timestamp & Block Number Dependency (TBD) | P8 | 1,951 |
| S7 | Incorrect Auth. Check (IAC) | P4 | 9,384 |
| S8 | Invalid Input Data / Failed Send (IID) | P3 | 839 |

---

## Full Dataset Reproduction

The full opcode sequence dataset can be reproduced using the following two open-source tools:

### Tool 1: TxSpector

TxSpector replays Ethereum mainnet transactions and extracts the EVM execution trace in the following 3-tuple format:

```
{<PC>; <OPCODE>; <ARGS>}
```

- **Repository**: https://github.com/OSUSecLab/TxSpector
- **Input**: Transaction hashes from `dataset_txhash_label.csv`
- **Output**: Raw execution trace files per transaction

### Tool 2: SODA (Smart-contract Online Detection and Analysis)

SODA's detection plugins monitor runtime behavior and assign vulnerability labels.

- **Repository**: https://github.com/pandabox-dev/SODA (original)
- **Plugin mapping used in this study**:
  - P1 → S1 (Re-entrancy)
  - P2 → S2 (Unexpected Function Invocation)
  - P5 → S3 (No Check After Invocation)
  - P6 → S4 (Missing Transfer Event)
  - P7 → S5 (Strict Check for Balance)
  - P8 → S6 (Timestamp Dependency)
  - P4 → S7 (Incorrect Auth. Check)
  - P3 → S8 (Invalid Input Data / Failed Send)

---

## Preprocessing Steps

After collecting raw traces via TxSpector + SODA, run the preprocessing script:

```bash
python code/preprocess.py
```

This script performs the following steps:

1. **PC and ARGS removal**: Extract only the OPCODE token from each 3-tuple
2. **Opcode normalization**:

   | Original | Normalized | Reason |
   |----------|------------|--------|
   | PUSH1–PUSH32 | PUSH | Operand size irrelevant to control flow |
   | DUP1–DUP16 | DUP | Stack depth irrelevant to operation semantics |
   | SWAP1–SWAP16 | SWAP | Stack depth irrelevant to operation semantics |
   | LOG0–LOG4 | LOG | Number of topics irrelevant to logging semantics |

3. **Sequence serialization**: Join normalized opcodes into a whitespace-delimited string
4. **Label assignment**: Map SODA plugin output to S0–S8 labels

The output is saved as a CSV with columns: `txhash`, `label`, `block`, `opcode_sequence`, `opcode_simplified`.

---

## Notes on Data Collection

- Transactions were collected from Ethereum mainnet blocks.
- Vulnerable transactions were identified by SODA plugins during transaction replay.
- Normal transactions (S0) were sampled from blocks with no detected vulnerabilities.
- S8 samples were added separately from the initial S1–S7 collection using SODA plugin P3.
- All transactions are publicly accessible on the Ethereum blockchain via block explorers (e.g., Etherscan) or archive nodes.

---

## Ethereum Node Requirement

Transaction replay requires access to an Ethereum **archive node** (full historical state).  
This can be set up using go-ethereum (Geth) with archive mode:

```bash
geth --syncmode full --gcmode archive --datadir /path/to/data
```

Alternatively, third-party archive node providers (e.g., Alchemy, Infura) can be used via JSON-RPC.
