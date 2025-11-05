"""
Log Ingest Attribute Analyzer

Description:
This script loads a JSON or CSV log sample file, analyzes its contents,
and provides insights into log attributes and potential high-volume anomalies.

NEW CAPABILITY:
--analyze_with_gemini
This flag will run all local analyses and then send a summary of the
findings to the Gemini API for a natural language summary and
advanced insight generation.

** REQUIRES --GEMINI_API_KEY to be set. **

Required Dependencies:
-   pandas: This is the only external library required.
    You can install it using pip:
    pip install pandas
-   numpy: A dependency of pandas.
-   requests: Used to call the Gemini API.
    You can install it using pip:
    pip install requests

Usage:
Save this file as 'attribute-analyzer.py' and run it from your terminal,
passing the path to your log sample file as an argument.

Example:
    python attribute-analyzer.py path/to/your-log-sample.csv

Advanced Usage:
    python attribute-analyzer.py path/to/sample.json --PRESENCE_THRESHOLD_PCT 50

Gemini Analysis:
    python attribute-analyzer.py path/to/sample.csv --analyze_with_gemini --GEMINI_API_KEY "YOUR_API_KEY"

"""

import json
import argparse
import pandas as pd
from collections import Counter
import os  # For file extension checking
import textwrap
import time # For status messages
import numpy as np # For checking nan
import requests # For making API calls

# --- Analysis Configuration ---
# (All previous constants remain here)
# ...
CARDINALITY_UPPER_LIMIT = 10000
BEST_COMBO_COUNT = 6
TOP_ANOMALOUS_MESSAGES = 5
MIN_FREQ_ANOMALY_THRESHOLD_PCT = 0.5
ATTRIBUTES_TO_IGNORE = [
    'timestamp', 
    'messageid', 
    'newrelic.logpattern'
]
PREFERRED_CONTEXT_ATTRIBUTES = [
    'level', 'log.level', 'severity',
    'logger', 'logger_name', 'filepath', 'file.path', 'plugin.source',
    'container_name', 'namespace_name', 'pod_name', 'cluster_name',
    'app', 'app.name', 'application', 'application.name', 'appName',
    'service', 'service.name', 'serviceName',
    'entity.name', 'entity.type',
    'env', 'environment',
    'host', 'hostname',
    'team', 'team.name', 'owner'
]
HASH_COLUMNS_TO_EXCLUDE = ATTRIBUTES_TO_IGNORE + [
    '@timestamp', 'newrelic.logs.batchIndex', 'origin.file.line',
    'trace.id', 'span.id', 'parent.id', 'traceid', 'spanid',
    'labels.pod-template-hash', 'labels.controller-revision-hash',
    'labels.apps.kubernetes.io/pod-index', 'labels.statefulset.kubernetes.io/pod-name',
    'pod_name', 'hostname', 'fullhostname',
    'instance.id', 'instance_id',
    'entity.guid', 'entity.guids',
    'xffheaderafterupdate', 'xffheaderoriginal', 'xffheaderoriginalvalues',
    'apigeemessageid', 'bl-correlationid', 'correlation.id',
    'request.id', 'request_id', 'messageId'
]


# --- Helper & Printing Functions ---
# (All previous helper functions: print_header, _dedup_names, load_log_file, etc.)
# ...
def print_header(title):
    """Prints a formatted section header."""
    print("\n" + ("-" * 60))
    print(f"### {title.upper()} ###")
    print(("-" * 60) + "\n")

def _dedup_names(names):
    """
    Robustly de-duplicates a list of names, appending .1, .2, etc.
    """
    names = list(names)
    counts = {}
    for i, name in enumerate(names):
        if name in counts:
            counts[name] += 1
            names[i] = f"{name}.{counts[name] - 1}"
        else:
            counts[name] = 1
    return names

def _get_metric_group(metric_name):
    """
    Applies logic to group a metric name by its prefix.
    e.g., 'aws.lambda.invocations' -> 'aws.lambda'
    e.g., 'http.server.requests' -> 'http.server'
    e.g., 'newrelic.apm.jvm.memory.heap.used' -> 'newrelic.apm'
    """
    try:
        parts = str(metric_name).split('.')
        if len(parts) <= 2:
            return str(metric_name) # Already a group or simple name
        
        # Define common high-level prefixes that benefit from 2-part grouping
        common_prefixes = ['aws', 'gcp', 'azure', 'newrelic', 'apm', 'http', 'db', 'database']
        
        if parts[0].lower() in common_prefixes and len(parts) > 2:
            # Return first two parts (e.g., 'aws.lambda', 'http.server')
            return f"{parts[0]}.{parts[1]}"
        else:
            # For unknown or simple prefixes, just return the first part
            return parts[0]
    except Exception:
        return 'unknown_group'

# --- CORRECTED HELPER ---
def find_first_column(df_columns, potential_names):
    """
    Finds the first column name in the DataFrame that exists from a list.
    This is case-insensitive as it assumes df_columns is already lowercased.
    """
    for name in potential_names:
        # --- FIX: check the lowercased version of the name ---
        if name.lower() in df_columns:
            return name.lower()
    return None
# --- END CORRECTED HELPER ---

def load_log_file(filepath):
    """
    Loads the log file (JSON or CSV) into a pandas DataFrame.
    This version now supports both JSON Arrays and JSON Lines files
    and prevents hangs by pre-flattening 'attributes' and using max_level=1.
    """
    # Status message is printed from main()
    df = None # Initialize df
    try:
        # Get file extension
        _root, file_ext = os.path.splitext(filepath)
        file_ext = file_ext.lower()

        if file_ext == '.json':
            print("  Detected JSON file. Reading file into memory...")
            data = []
            with open(filepath, 'r', encoding='utf-8') as f:
                # --- NEW: Try to load as array, fallback to JSON Lines ---
                try:
                    # Try loading as a single JSON array
                    data = json.load(f)
                    print("  File loaded as a single JSON array.")
                except json.JSONDecodeError as e:
                    # If it fails, rewind and try to read as JSON Lines
                    print(f"  ...Could not parse as JSON array ({e}). Checking for JSON Lines...")
                    f.seek(0) # Rewind file
                    lines = f.readlines()
                    for i, line in enumerate(lines):
                        if line.strip():
                            try:
                                data.append(json.loads(line))
                            except json.JSONDecodeError as line_e:
                                print(f"  ...Skipping malformed JSON line #{i+1}: {line_e}")
                    if not data:
                        print("  Error: File is not a valid JSON array or JSON Lines.")
                        return None
                    print("  File successfully loaded as JSON Lines.")
            
            print(f"  Loaded {len(data)} JSON objects. Normalizing into DataFrame...")
            if not isinstance(data, list):
                print("  Error: Loaded data is not a list (this shouldn't happen).")
                return None
            
            # --- NEW: Pre-flatten the 'attributes' field ---
            print("  ...Pre-flattening 'attributes' field for performance...")
            processed_data = []
            count_flattened = 0
            for log in data:
                if 'attributes' in log and isinstance(log['attributes'], dict):
                    # Pop the attributes dict to modify it
                    attrs = log.pop('attributes')
                    for key, value in attrs.items():
                        log[f'attributes.{key}'] = value
                    count_flattened += 1
                processed_data.append(log)
            
            if count_flattened > 0:
                print(f"  ...Promoted 'attributes' fields in {count_flattened} logs.")
            data = processed_data # Overwrite data with the processed version
            # --- END PRE-FLATTENING ---
            
            # --- FIX FOR HANGING ---
            # Set max_level=1 as requested, since we pre-flattened 'attributes'.
            print("  ...Flattening all other fields (max_level=1).")
            df = pd.json_normalize(data, max_level=1) 
            # --- END FIX ---

        elif file_ext == '.csv':
            print("  Detected CSV file. Reading file into DataFrame...")
            # Removed 'mangle_dupe_cols' - it's default in pandas 2.0+
            df = pd.read_csv(filepath) 
            
            if df.empty:
                print("  Warning: CSV file is empty.")

        else:
            print(f"  Error: Unsupported file type '{file_ext}'. "
                  "Only .json and .csv files are supported.")
            return None
        
        # --- Normalize all column names ---
        if df is None or df.empty:
            print("  Warning: File loaded, but no data was found.")
            return df # Return the empty dataframe
        
        print("  Normalizing column names (lowercase and stripping spaces)...")
        df.columns = df.columns.str.lower().str.strip()
        
        # --- FIX for Duplicated Normalized Columns ---
        if df.columns.has_duplicates:
            print("  Found columns that are duplicates after normalization.")
            print("  Renaming duplicates (e.g., 'level.1', 'level.2')...")
            df.columns = _dedup_names(df.columns)
        # --- END FIX ---
        
        print(f"  Successfully loaded {len(df)} log entries.")
        return df

    except FileNotFoundError:
        print(f"  Error: File not found at '{filepath}'")
        return None
    except json.JSONDecodeError:
        # This will catch a file that is not JSON at all
        print("  Error: Could not decode JSON. File may be corrupt or not valid JSON.")
        return None
    except pd.errors.ParserError:
        print(f"  Error: Could not parse CSV. File may be malformed.")
        return None
    except UnicodeDecodeError:
        print("  Error: Could not read file due to a UnicodeDecodeError. "
              "Please ensure the file is UTF-8 encoded.")
        return None
    except Exception as e:
        print(f"  An unexpected error occurred during file loading: {e}")
        return None

def analyze_attributes(df):
    """
    Analyzes each attribute (column) in the DataFrame for presence,
    cardinality, and examples.
    """
    print("  Analyzing attributes (this may take a moment on large files)...")
    start_time = time.time()
    total_logs = len(df)
    attribute_stats = []
    
    # --- Calculate total summed length ---
    print("  ...Calculating total data size (this may take a moment)...")
    total_summed_length = 0
    # Create a list of length Series to avoid re-calculation
    all_lengths = {}
    for col in df.columns:
        # Store lengths for every column
        lengths = df[col].astype(str).str.len()
        all_lengths[col] = lengths
        total_summed_length += lengths.sum()
    
    if total_summed_length == 0:
        total_summed_length = 1  # Avoid division by zero
    
    print(f"  ...Total data size in sample: {total_summed_length} characters.")
    # --- END ---

    # load_log_file guarantees unique columns
    for col in df.columns:
        # Check base name (e.g., 'level' for 'level.1')
        base_col = col.split('.')[0]
        if col in ATTRIBUTES_TO_IGNORE or base_col in ATTRIBUTES_TO_IGNORE:
            continue
        
        col_data = df[col]
        non_null_count = col_data.count()

        if non_null_count == 0:
            continue

        presence_pct = (non_null_count / total_logs) * 100
        unique_values = col_data.nunique()
        examples = col_data.dropna().unique()[:5]
        
        # --- Get new length stats ---
        lengths = all_lengths[col] # Get pre-calculated lengths
        
        # Calculate stats only on non-null values
        # We must re-filter 'lengths' to only non-nulls for accurate percentiles
        non_null_lengths = lengths[col_data.notna()]
        
        if non_null_lengths.empty:
            # Handle columns that are all null
            max_length = 0
            p50_length = 0
            p90_length = 0
            col_summed_length = 0
        else:
            max_length = non_null_lengths.max()
            p50_length = non_null_lengths.quantile(0.5)
            p90_length = non_null_lengths.quantile(0.9)
            # col_summed_length uses all lengths (nulls were converted to "nan")
            col_summed_length = lengths.sum()

        contribution_pct = (col_summed_length / total_summed_length) * 100
        # --- END ---
        
        attribute_stats.append({
            "attribute": col,
            "presence_count": non_null_count,
            "presence_pct": presence_pct,
            "unique_values": unique_values,
            "examples": list(examples),
            # --- Add new stats to dict ---
            "max_length": max_length,
            "p50_length": p50_length,
            "p90_length": p90_length,
            "contribution_pct": contribution_pct
            # --- END ---
        })

    # Sort by presence (descending), then unique values (ascending)
    # We will re-sort this by contribution_pct in the print function
    sorted_stats = sorted(attribute_stats,
                          key=lambda x: (-x['presence_pct'], x['unique_values']))
    
    end_time = time.time()
    print(f"  ...Attribute analysis complete ({end_time - start_time:.2f}s).")
    return total_logs, sorted_stats

def print_best_attributes(total_logs, sorted_stats, presence_threshold_pct):
    """
    Prints the formatted analysis for the "Best Attributes" section.
    """
    print_header(f"Log Sample Count: {total_logs}")

    print_header("Attribute Analysis for Ingest Subdivision")
    print(f"Finding attributes with > {presence_threshold_pct}% presence "
          f"and < {CARDINALITY_UPPER_LIMIT} unique values.")
    print("Attributes are sorted by their **total size contribution** (descending).")

    best_attributes = []
    found_attributes = False
    
    # --- Filter first, then re-sort by contribution ---
    filtered_stats = []
    for item in sorted_stats:
        # Filter for attributes that are good candidates for subdivision
        if (item['presence_pct'] >= presence_threshold_pct and
            item['unique_values'] > 1 and
            item['unique_values'] <= CARDINALITY_UPPER_LIMIT):
            
            filtered_stats.append(item)
            best_attributes.append(item['attribute']) # Add to best for combo
            
    # Now, sort the filtered list by contribution
    sorted_by_contribution = sorted(filtered_stats,
                                    key=lambda x: -x['contribution_pct'])
    # --- END ---

    for item in sorted_by_contribution:
        found_attributes = True
        print("\n" + ("-" * 20))
        print(f"**{item['attribute']}**")
        
        # --- Add new stats to output ---
        print(f"    * **Total Size Contribution:** {item['contribution_pct']:.2f}%")
        print(f"    * **Presence:** {item['presence_pct']:.1f}% "
              f"({item['presence_count']} out of {total_logs} logs)")
        print(f"    * **Unique Values:** {item['unique_values']}")
        print(f"    * **Max Length:** {item['max_length']:.0f} chars")
        print(f"    * **50th Percentile Length:** {item['p50_length']:.0f} chars")
        print(f"    * **90th Percentile Length:** {item['p90_length']:.0f} chars")
        # --- END ---
            
        # Format examples for clean printing
        example_str = ", ".join([f'"{str(e)}"' for e in item['examples']])
        print(f"    * **Examples:** {textwrap.shorten(example_str, width=300, placeholder=' (truncated)...')}")

    if not found_attributes:
        print(f"\nNo attributes met the {presence_threshold_pct}% presence threshold.")

    # Return the original best_attributes list, which is sorted by presence/cardinality
    # This is better for the *combination* analysis
    return best_attributes

def print_combination_analysis(best_attributes):
    """
    Prints the combination analysis and a sample NRQL query.
    """
    print_header("Attribute Combination Analysis")

    if not best_attributes:
        print("No best attributes found, skipping combination analysis.")
        return

    # Select the top N attributes for the combination
    # This list is sorted by presence, which is good for faceting
    combo_attributes = best_attributes[:BEST_COMBO_COUNT]
    
    print(f"The following {len(combo_attributes)} attributes (selected by *presence* and *cardinality*) "
          "provide a strong, multi-dimensional breakdown of log ingest:")
    for i, attr in enumerate(combo_attributes, 1):
        print(f"    {i}. **{attr}**")

    print("\n#### Example NRQL Query ####")
    print("Use this query in New Relic to see your log counts faceted "
          "by this combination:")
    
    # Need to quote attributes that have a '.'
    facet_list = [f"`{attr}`" if '.' in attr else attr for attr in combo_attributes]
    facet_string = ", ".join(facet_list)
    
    nrql_query = f"SELECT count(*) FROM Log \nFACET {facet_string} \nSINCE 1 HOUR AGO"
    print("\n```nrql\n" + nrql_query + "\n```")

def infer_anomaly_type(message, level):
    """
    Analyzes the message and level to classify the anomaly.
    Returns (anomaly_type, insight_description)
    """
    message_lower = str(message).lower()
    level_lower = str(level).lower()

    # --- Type 1: Repetitive Error / Failure ---
    error_keywords = ['failed', 'failure', 'error', 'exception', 'cannot connect', 'temporary bulk send failure']
    if level_lower in ['error', 'warn', 'fatal', 'critical'] or any(kw in message_lower for kw in error_keywords):
        return (
            "Potential Log Storm (Repetitive Error)",
            "This indicates a component is likely stuck in a retry loop (e.g., cannot "
            "connect to a destination) OR simply a repetitive logger. Fixing the root cause will stop this log storm."
        )

    # --- Type 2: Repetitive "OK" / Polling ---
    polling_keywords = ['health', 'status requested', 'skipping patch', 'no status changes', 'check-in', 'success', 'started call', 'finished call']
    if any(kw in message_lower for kw in polling_keywords) and level_lower in ['info', 'n/a', 'debug']:
        return (
            "Potential Low-Value Polling/Health Check",
            "This appears to be a repetitive 'check-in' or health check log. "
            "These are often safe to filter at the source or lower to a DEBUG level."
        )
    
    # --- Type 3: Verbose Informational Log ---
    # This is the catch-all for high-frequency info/debug logs
    if level_lower in ['info', 'debug', 'n/a']:
        return (
            "Potential Verbose 'Chatter' Log",
            "This is a high-frequency informational log. It may be logging a common "
            "action on every request or loop, which can often be sampled or lowered in severity."
        )
    
    # --- Fallback ---
    return (
        "High-Frequency Combination",
        "The specific reason is unclear, but this combination of attributes "
        "and message is extremely frequent in the sample and warrants investigation."
    )

def calculate_log_hashes_and_size(df, payload_size_percentile):
    """
    Calculates log hashes and total payload size for each row.
    Adds 'log_hash' and 'log_total_size' to the DataFrame.
    Returns the size threshold for large payloads.
    """
    print("\n" + ("=" * 20))
    print("  Starting Log Hash and Size calculation...")
    start_time = time.time()
    
    # Find the set of columns to hash
    exclude_cols = set(HASH_COLUMNS_TO_EXCLUDE)
    cols_to_hash = [col for col in df.columns if col not in exclude_cols]
    
    hash_failed = False
    
    # --- Find message column (could be flat or nested) ---
    # We need to hash 'message' if it exists, but not 'message' if it's JSON
    # For hashing, we'll just use the raw 'message' field if it exists.
    # A proper hash would parse the JSON message, remove IDs, and re-serialize
    # but that is too slow. We will rely on 'message' column.
    
    message_col = find_first_column(df.columns, ['message'])
    
    if not message_col:
        print("  ...Skipping hash analysis: 'message' column not found.")
        hash_failed = True
    else:
        # Ensure message is in the hash list if it's not excluded
        if message_col not in cols_to_hash and message_col not in exclude_cols:
             cols_to_hash.append(message_col)
             
        print(f"  ...Hashing based on {len(cols_to_hash)} attributes (including '{message_col}').")
        try:
            # Convert all columns to string before hashing to avoid errors
            hashes = pd.util.hash_pandas_object(df[cols_to_hash].astype(str), index=False)
            df['log_hash'] = hashes
        except Exception as e:
            print(f"  ...An error occurred during log hashing: {e}")
            hash_failed = True

    # Calculate total size of all string-converted columns
    print("  ...Calculating total payload size for each log.")
    try:
        # Sum the length of all columns (as strings) for each row
        df['log_total_size'] = df.astype(str).apply(lambda x: x.str.len()).sum(axis=1)
        size_threshold = df['log_total_size'].quantile(payload_size_percentile)
        
    except Exception as e:
        print(f"  ...An error occurred during size calculation: {e}")
        return None, None

    end_time = time.time()
    if hash_failed:
        print(f"  ...Size calculation complete ({end_time - start_time:.2f}s).")
    else:
        print(f"  ...Hash and size calculation complete ({end_time - start_time:.2f}s).")
    
    print(f"  ...Top {100*(1-payload_size_percentile):.1f}% payload size threshold: {size_threshold:.0f} chars.")
    
    return size_threshold, df

def print_duplicate_log_hash_anomalies(df, total_logs, log_hash_frequency_threshold):
    """
    Finds functionally identical logs by hashing rows (omitting IDs).
    Assumes 'log_hash' column already exists on df.
    """
    print("\n" + ("=" * 20))
    print("  Starting Duplicate Log Hash analysis...")
    start_time = time.time()
    
    if 'log_hash' not in df.columns:
        print("  ...Skipping duplicate hash analysis: 'log_hash' column not found.")
        return
        
    try:
        hash_counts = df['log_hash'].value_counts()
    except Exception as e:
        print(f"  ...An error occurred during hash counting: {e}")
        return

    # Filter by the frequency threshold
    min_count_threshold = total_logs * log_hash_frequency_threshold
    
    frequent_hashes = hash_counts[hash_counts >= min_count_threshold]

    if frequent_hashes.empty:
        print(f"  ...No duplicate log hashes found that meet the {log_hash_frequency_threshold*100:.1f}% "
              "sample threshold.")
        end_time = time.time()
        print(f"  ...Duplicate log hash analysis complete ({end_time - start_time:.2f}s).")
        return

    print(f"\nFound {len(frequent_hashes)} types of duplicate logs "
          f"(that meet the {log_hash_frequency_threshold*100:.1f}% threshold).\n")

    # Get the *base* context columns that exist in the DF
    base_context_cols = [col for col in PREFERRED_CONTEXT_ATTRIBUTES if col in df.columns]
    message_col = find_first_column(df.columns, ['message'])

    for i, (hash_val, count) in enumerate(frequent_hashes.head(TOP_ANOMALOUS_MESSAGES).items(), 1):
        
        # Get the very first row that matches this hash
        first_row = df.loc[df['log_hash'] == hash_val].iloc[0]
        
        message = first_row.get(message_col, 'N/A')
        level_key = next((k for k in first_row.index if k in ['level', 'log.level', 'severity']), None)
        level = first_row.get(level_key, 'N/A') if level_key else 'N/A'

        anomaly_type, anomaly_desc = infer_anomaly_type(message, level)

        print(f"**Duplicate Log Anomaly #{i}**")
        print(f"    * **Count in Sample:** {count} "
              f"({(count / total_logs * 100):.1f}% of sample)")
        print(f"    * **Anomaly Type:** {anomaly_type}")
        print(f"    * **Insight:** {anomaly_desc} This *exact* log (minus IDs/timestamps) "
              f"was found {count} times.")

        print("    * **Example Log Context:**")
        
        # --- MODIFICATION: Only print attributes that exist (are not NaN) ---
        context_cols_to_print = [message_col] + base_context_cols
        for col_name in context_cols_to_print:
            if col_name and col_name in first_row.index: # Check if col_name is not None
                value = first_row.get(col_name)
                # Check if value is NaN or None
                if pd.isna(value):
                    continue # Skip printing this attribute
                    
                value_str = str(value)
                if len(value_str) > 70 and col_name != message_col: 
                    value_str = value_str[:70] + "..."
                elif len(value_str) > 150: # Longer truncation for message
                    value_str = value_str[:150] + "..."
                    
                print(f"        - {col_name}: \"{value_str}\"")
        # --- END MODIFICATION ---

        print("-" * 20)
        
    if len(frequent_hashes) > TOP_ANOMALOUS_MESSAGES:
        print(f"...and {len(frequent_hashes) - TOP_ANOMALOUS_MESSAGES} more duplicate log types found.")

    end_time = time.time()
    print(f"  ...Duplicate log hash analysis complete ({end_time - start_time:.2f}s).")

def print_large_payload_hash_anomalies(df, total_logs, size_threshold, hash_freq_threshold):
    """
    Finds logs that are both large (>= size_threshold) and frequent
    (>= hash_freq_threshold).
    Assumes 'log_hash' and 'log_total_size' columns exist.
    """
    print("\n" + ("=" * 20))
    print(f"  Starting Large Payload Hash analysis...")
    start_time = time.time()
    
    if 'log_hash' not in df.columns or 'log_total_size' not in df.columns:
        print("  ...Skipping large payload analysis: hash/size columns not found.")
        return

    try:
        # 1. Filter for logs that are "large"
        large_logs = df[df['log_total_size'] >= size_threshold]
        
        if large_logs.empty:
            print(f"  ...No logs found at or above the size threshold of {size_threshold:.0f} chars.")
            end_time = time.time()
            print(f"  ...Large payload hash analysis complete ({end_time - start_time:.2f}s).")
            return
            
        # 2. Count the frequency of these large log hashes
        hash_counts = large_logs['log_hash'].value_counts()
        
        # 3. Filter for hashes that are frequent *relative to the total sample*
        min_count_threshold = total_logs * hash_freq_threshold
        frequent_large_hashes = hash_counts[hash_counts >= min_count_threshold]

    except Exception as e:
        print(f"  ...An error occurred during large payload analysis: {e}")
        return

    if frequent_large_hashes.empty:
        print(f"  ...No *frequent* large payloads found (threshold: {hash_freq_threshold*100:.1f}% frequency).")
        end_time = time.time()
        print(f"  ...Large payload hash analysis complete ({end_time - start_time:.2f}s).")
        return

    print(f"\nFound {len(frequent_large_hashes)} types of *Large & Frequent* logs.\n")

    # Get the *base* context columns that exist in the DF
    base_context_cols = [col for col in PREFERRED_CONTEXT_ATTRIBUTES if col in df.columns]
    message_col = find_first_column(df.columns, ['message'])

    for i, (hash_val, count) in enumerate(frequent_large_hashes.head(TOP_ANOMALOUS_MESSAGES).items(), 1):
        
        # Get the first row for this hash
        first_row = df.loc[df['log_hash'] == hash_val].iloc[0]
        
        message = first_row.get(message_col, 'N/A')
        level_key = next((k for k in first_row.index if k in ['level', 'log.level', 'severity']), None)
        level = first_row.get(level_key, 'N/A') if level_key else 'N/A'

        anomaly_type, anomaly_desc = infer_anomaly_type(message, level)

        print(f"**Large Payload Anomaly #{i}**")
        print(f"    * **Count in Sample:** {count} "
              f"({(count / total_logs * 100):.1f}% of sample)")
        print(f"    * **Payload Size:** {first_row['log_total_size']} characters")
        print(f"    * **Anomaly Type:** Large Payload ({anomaly_type})")
        
        # Calculate percentile rank for this specific log
        rank = df['log_total_size'].rank(pct=True).loc[first_row.name] * 100
        
        print(f"    * **Insight:** This log is both *very large* (in the top "
              f"{100 - rank:.1f}% of payloads) and "
              f"*very frequent*. This is a primary target for cost reduction.")

        print("    * **Example Log Context:**")
        
        # --- MODIFICATION: Only print attributes that exist (are not NaN) ---
        context_cols_to_print = [message_col] + base_context_cols
        for col_name in context_cols_to_print:
            if col_name and col_name in first_row.index: # Check if col_name is not None
                value = first_row.get(col_name)
                # Check if value is NaN or None
                if pd.isna(value):
                    continue # Skip printing this attribute
                    
                value_str = str(value)
                if len(value_str) > 70 and col_name != message_col: 
                    value_str = value_str[:70] + "..."
                elif len(value_str) > 150: # Longer truncation for message
                    value_str = value_str[:150] + "..."
                    
                print(f"        - {col_name}: \"{value_str}\"")
        # --- END MODIFICATION ---

        print("-" * 20)
        
    if len(frequent_large_hashes) > TOP_ANOMALOUS_MESSAGES:
        print(f"...and {len(frequent_large_hashes) - TOP_ANOMALOUS_MESSAGES} more large payload types found.")

    end_time = time.time()
    print(f"  ...Large payload hash analysis complete ({end_time - start_time:.2f}s).")


def print_high_frequency_anomalies(df, total_logs, top_n):
    """
    Finds and prints the most frequent, repetitive log messages
    combined with other key contextual attributes.
    """
    print("\n" + ("=" * 20))
    print("  Starting High-Frequency Message analysis...")
    print("  (This finds similar messages, ignoring pod names/IDs)")
    start_time = time.time()

    message_col = find_first_column(df.columns, ['message'])

    # All columns are normalized, so just check for 'message'
    if not message_col:
        print("  Cannot perform frequency analysis: 'message' column not found.")
        print(f"  Available columns are: {list(df.columns)}")
        return

    # Find which context attributes are present in the df
    # For this analysis, we *don't* want instance-specific fields
    # so we'll use a modified list
    context_cols_to_use = [
        col for col in PREFERRED_CONTEXT_ATTRIBUTES 
        if col in df.columns and col not in [
            'pod_name', 'hostname', 'fullhostname', 'filepath', 'file.path'
        ]
    ]
    
    group_by_cols = [message_col] + context_cols_to_use

    print(f"  Analyzing anomalies by grouping: {group_by_cols}")

    try:
        analysis_df = df[group_by_cols].copy()
        
        # Convert message to string and strip whitespace
        analysis_df[message_col] = analysis_df[message_col].astype(str).str.strip()
        
        # Fill NaN in context columns so they are grouped as 'N/A'
        for col in group_by_cols:
            if col != message_col:
                analysis_df[col] = analysis_df[col].fillna('N/A')

        combination_counts = analysis_df.groupby(group_by_cols).size()
        top_combinations = combination_counts.sort_values(ascending=False)

    except Exception as e:
        print(f"  An error occurred during frequency grouping: {e}")
        return

    if top_combinations.empty:
        print("  No message combinations found to analyze for anomalies.")
        return
    
    end_time = time.time()
    print(f"  ...Message analysis complete ({end_time - start_time:.2f}s).")

    # --- Filter by frequency threshold ---
    min_count_threshold = total_logs * (MIN_FREQ_ANOMALY_THRESHOLD_PCT / 100.0)
    
    final_anomalies = top_combinations[top_combinations >= min_count_threshold]
    
    if final_anomalies.empty:
        print(f"\nNo high-frequency message anomalies found that meet the {MIN_FREQ_ANOMALY_THRESHOLD_PCT}% "
              "sample threshold.")
        return
        
    print(f"\nShowing the Top {top_n} most frequent log *message combinations* "
          f"(that meet the {MIN_FREQ_ANOMALY_THRESHOLD_PCT}% threshold).\n")

    for i, (combination, count) in enumerate(final_anomalies.head(top_n).items(), 1):
        
        print(f"**Frequency Anomaly #{i}**")
        print(f"    * **Count in Sample:** {count} "
              f"({(count / total_logs * 100):.1f}% of sample)")
        
        # --- Anomaly Classification ---
        message = "N/A"
        level = "N/A"
        
        if isinstance(combination, tuple):
            # Create a dict from the combination for easy lookup
            combo_dict = dict(zip(group_by_cols, combination))
            message = combo_dict.get(message_col, "N/A")
            
            # Find the level, checking for 'level' or other common names
            level_key = next((k for k in combo_dict if k in ['level', 'log.level', 'severity']), None)
            if level_key:
                level = combo_dict[level_key]
        else:
             # Case for only 'message'
             message = combination
        
        anomaly_type, anomaly_desc = infer_anomaly_type(message, level)
        
        print(f"    * **Anomaly Type:** {anomaly_type}")
        print(f"    * **Insight:** {anomaly_desc}")
        # --- END ---

        print("    * **Combination:**")
        
        if isinstance(combination, tuple):
            for col_name, value in zip(group_by_cols, combination):
                # --- MODIFICATION: Only print attributes that are not "N/A" ---
                if value != 'N/A':
                    value_str = str(value)
                    if len(value_str) > 150: # Truncate long messages
                        value_str = value_str[:150] + "..."
                    print(f"        - {col_name}: \"{value_str}\"")
                # --- END MODIFICATION ---
        else:
             value_str = str(combination)
             if len(value_str) > 150:
                 value_str = value_str[:150] + "..."
             print(f"        - {message_col}: \"{value_str}\"")

        print("-" * 20)

def print_large_attribute_anomalies(df, total_logs, large_attr_char_length, large_attr_percentile, large_attr_presence_threshold):
    """
    Finds attributes that are consistently storing large string values.
    """
    print("\n" + ("=" * 20))
    print("  Starting large attribute analysis...")
    start_time = time.time()
    
    message_col = find_first_column(df.columns, ['message'])
    
    # Don't check 'message' or other known-large fields
    EXCLUDE_FROM_LARGE_CHECK = ATTRIBUTES_TO_IGNORE + [message_col]
    
    found_anomalies = False
    
    for col in df.columns:
        if col in EXCLUDE_FROM_LARGE_CHECK:
            continue
            
        # Only check string-like data
        try:
            # Convert to string, calculate length, ignore errors
            lengths = df[col].astype(str).str.len()
        except:
            continue # Not a string-like column
            
        # Check presence
        non_null_count = df[col].count()
        if (non_null_count / total_logs) < large_attr_presence_threshold:
            continue # Skip low-presence attributes
            
        # Get the percentile length
        try:
            percentile_length = lengths.quantile(large_attr_percentile)
        except Exception:
            continue # Error in quantile calculation

        if percentile_length >= large_attr_char_length:
            if not found_anomalies:
                print(f"\nFound attributes with consistently large values (>= {large_attr_char_length} chars).\n")
            found_anomalies = True
            
            # Get a non-null example
            example = df[col].dropna().iloc[0]
            example_str = str(example)
            if len(example_str) > 150:
                example_str = example_str[:150] + "..."
                
            print(f"**Large Attribute Anomaly: `{col}`**")
            print(f"    * **Anomaly Type:** Potential Payload/Stack Trace Storage")
            print(f"    * **Insight:** {int(large_attr_percentile * 100)}% of values for this attribute are "
                  f"~{int(percentile_length)} characters or longer. This suggests "
                  "it may be storing large payloads or stack traces, which can "
                  "significantly increase ingest size.")
            print(f"    * **Example (truncated):** \"{example_str}\"")
            print("-" * 20)
            
    if not found_anomalies:
        print(f"  ...No consistently large attributes found (threshold: >={large_attr_char_length} "
              f"chars at {int(large_attr_percentile * 100)}th percentile).")
        
    end_time = time.time()
    print(f"  ...Large attribute analysis complete ({end_time - start_time:.2f}s).")

def print_truncated_log_anomalies(df, total_logs):
    """
    Finds log messages that end in a newline, indicating broken multi-line logs.
    """
    print("\n" + ("=" * 20))
    print("  Starting truncated log analysis...")
    start_time = time.time()
    
    message_col = find_first_column(df.columns, ['message'])
    
    if not message_col:
        print("  ...Cannot perform truncated log analysis: 'message' column not found.")
        return

    try:
        # Find all non-null messages that are strings and end with \n
        truncated_logs = df[df[message_col].astype(str).str.endswith('\n', na=False)]
        count = len(truncated_logs)
    except Exception as e:
        print(f"  ...An error occurred during truncated log analysis: {e}")
        return

    if count > 0:
        percentage = (count / total_logs) * 100
        example = truncated_logs.iloc[0][message_col]
        if len(example) > 200:
            example = "..." + example[-200:] # Show the end of the line
        
        print(f"\n**Truncated Log Anomaly: `message`**")
        print(f"    * **Anomaly Type:** Broken Multi-Line Log")
        print(f"    * **Count in Sample:** {count} ({percentage:.1f}% of sample)")
        print(f"    * **Insight:** These logs end in a newline ('\\n'), which strongly "
              "suggests multi-line logs (like stack traces) are being broken and "
              "sent as separate entries. This inflates log counts and breaks "
              "parsing. This can often be fixed in your log forwarder's "
              "multi-line configuration.")
        print(f"    * **Example (showing end of line):** \"{example}\"")
        print("-" * 20)
    else:
        print("  ...No truncated (newline-terminated) logs found.")

    end_time = time.time()
    print(f"  ...Truncated log analysis complete ({end_time - start_time:.2f}s).")

def print_all_anomaly_insights(df_with_hashes, size_threshold, total_logs, top_n, 
                               log_hash_frequency_threshold,
                               payload_size_hash_frequency,
                               large_attr_char_length, 
                               large_attr_percentile, 
                               large_attr_presence_threshold):
    """
    Master function to run all types of anomaly analysis.
    Assumes df_with_hashes already contains 'log_hash' and 'log_total_size'.
    """
    print_header("Potential Anomaly Insights")
    
    # --- MODIFICATION: Hashing/Sizing is now done in main() ---
    
    if size_threshold is None or df_with_hashes is None or 'log_hash' not in df_with_hashes.columns:
        print("...Skipping all hash-based anomaly detection (required columns not found).")
    else:
        # 2. Duplicate Log Hash Analysis (Uses 'log_hash')
        # We pass a copy so the drops inside don't affect other analyses
        print_duplicate_log_hash_anomalies(df_with_hashes.copy(), total_logs, log_hash_frequency_threshold)
    
        # 3. Large Payload Hash Analysis (Uses 'log_hash' and 'log_total_size')
        print_large_payload_hash_anomalies(df_with_hashes.copy(), total_logs, size_threshold, payload_size_hash_frequency)
    
    # 4. High-Frequency Message Combinations (Message-based)
    # This function doesn't need the hash columns
    print_high_frequency_anomalies(df_with_hashes, total_logs, top_n)
    
    # 5. Large Attribute Analysis
    print_large_attribute_anomalies(df_with_hashes, total_logs, large_attr_char_length, large_attr_percentile, large_attr_presence_threshold)
    
    # 6. Truncated Log Analysis
    print_truncated_log_anomalies(df_with_hashes, total_logs)
    # --- END MODIFICATION ---

# --- MODIFIED FUNCTION ---
def print_metricname_deep_dive(df_with_metrics, total_logs):
    """
    Performs a deep-dive analysis on 'metricName' if it exists,
    breaking down ingest by metric, group, and newrelic.source.
    
    This new version reports on BOTH ingest size and log count
    and can find metrics at the top-level OR inside a JSON 'message' string.
    """
    print_header("MetricName Deep Dive Analysis")
    
    df_columns = df_with_metrics.columns
    
    # --- Define a helper to safely parse JSON from a string ---
    def _try_parse_json(val, key_to_get):
        if not isinstance(val, str) or not val.startswith('{'):
            return None
        try:
            data = json.loads(val)
            # Handle nested 'attributes' key inside message
            if 'attributes' in data and isinstance(data['attributes'], dict):
                return data['attributes'].get(key_to_get)
            return data.get(key_to_get)
        except json.JSONDecodeError:
            return None
    
    # --- Find all possible columns ---
    flat_metric_col = find_first_column(df_columns, ['metricname'])
    message_col = find_first_column(df_columns, ['message'])
    
    flat_source_col = find_first_column(df_columns, ['newrelic.source'])
    logtype_col = find_first_column(df_columns, ['logtype'])

    # --- DEBUGGING: Print found columns ---
    # print("  ...[Debug] Checking for metric/source columns:")
    # print(f"  ...Found 'metricname': {flat_metric_col}")
    # print(f"  ...Found 'message': {message_col}")
    # print(f"  ...Found 'newrelic.source': {flat_source_col}")
    # print(f"  ...Found 'logtype': {logtype_col}")
    # --- END DEBUGGING ---

    # --- Create a new DataFrame and COALESCE the columns ---
    metric_df = df_with_metrics[['log_total_size']].copy()
    final_metric_col = 'final_metric_name'
    final_source_col = 'final_source'
    
    # --- Coalesce metric columns ---
    metric_df[final_metric_col] = np.nan
    # 1. Fill with metrics from flat 'metricname' column (for JFR)
    if flat_metric_col:
        metric_df[final_metric_col] = metric_df[final_metric_col].fillna(df_with_metrics[flat_metric_col])
    # 2. Fill with metrics found inside JSON 'message' string (for AWS/GCP/Azure)
    if message_col:
        print("  ...Parsing 'message' field for JSON-embedded metrics...")
        message_metrics = df_with_metrics[message_col].apply(_try_parse_json, key_to_get='metricName')
        metric_df[final_metric_col] = metric_df[final_metric_col].fillna(message_metrics)

    # --- Coalesce source columns (in priority order) ---
    metric_df[final_source_col] = np.nan
    # 1. Fill with 'newrelic.source' from inside JSON 'message' (for AWS)
    if message_col:
        print("  ...Parsing 'message' field for JSON-embedded sources...")
        message_sources = df_with_metrics[message_col].apply(_try_parse_json, key_to_get='newrelic.source')
        metric_df[final_source_col] = metric_df[final_source_col].fillna(message_sources)
    # 2. Fill with flat 'newrelic.source' (for JFR)
    if flat_source_col:
        metric_df[final_source_col] = metric_df[final_source_col].fillna(df_with_metrics[flat_source_col])
    # 3. Fill with 'logtype' (for GCP/Azure)
    if logtype_col:
        metric_df[final_source_col] = metric_df[final_source_col].fillna(df_with_metrics[logtype_col])
    # 4. Fill any remaining with 'N/A'
    metric_df[final_source_col] = metric_df[final_source_col].fillna('N/A')
    
    # --- END COALESCE ---

    # Now, filter the new DataFrame to only rows that have a metric name
    metric_df = metric_df[metric_df[final_metric_col].notna()]
    
    if metric_df.empty:
        print(f"  ...No metrics found in 'metricname' or inside 'message' fields. Skipping.")
        return None

    # Clean data
    metric_df[final_metric_col] = metric_df[final_metric_col].astype(str)
    metric_df[final_source_col] = metric_df[final_source_col].astype(str)
    metric_df['metric_group'] = metric_df[final_metric_col].apply(_get_metric_group)
    
    total_metric_ingest = metric_df['log_total_size'].sum()
    total_metric_count = len(metric_df)
    total_sample_ingest = df_with_metrics['log_total_size'].sum()
    
    if total_sample_ingest == 0: total_sample_ingest = 1 # Avoid div by zero
    if total_metric_ingest == 0: total_metric_ingest = 1
    if total_metric_count == 0: total_metric_count = 1

    print(f"  ...Found {total_metric_count} logs with metric data (from all sources).")
    print(f"  ...These logs contribute {total_metric_ingest} chars "
          f"({(total_metric_ingest / total_sample_ingest * 100):.1f}%) of total sample ingest.")

    # String builder for Gemini summary
    gemini_summary = [
        f"Metric logs make up {total_metric_count} of {total_logs} logs ({total_metric_count/total_logs*100:.1f}%) "
        f"and contribute {(total_metric_ingest / total_sample_ingest * 100):.1f}% of total ingest size."
    ]

    # --- Run all 3 groupings using agg for both sum and count ---
    source_stats = metric_df.groupby(final_source_col).agg(
        total_size=('log_total_size', 'sum'),
        total_count=('log_total_size', 'size')
    )
    
    metric_stats = metric_df.groupby([final_metric_col, final_source_col]).agg(
        total_size=('log_total_size', 'sum'),
        total_count=('log_total_size', 'size')
    )
    
    group_stats = metric_df.groupby(['metric_group', final_source_col]).agg(
        total_size=('log_total_size', 'sum'),
        total_count=('log_total_size', 'size')
    )

    # === SECTION 1: BY INGEST SIZE (COST) ===
    print("\n" + ("=" * 20))
    print("  ### Breakdown by Ingest Size (Cost Analysis) ###")
    print("=" * 20)

    # --- By Source (Size) ---
    print("\n  **By Source (Sorted by Size):**")
    gemini_summary.append("\nTop 3 Sources by Size:")
    for source, row in source_stats.sort_values(by='total_size', ascending=False).head(5).iterrows():
        pct = (row['total_size'] / total_metric_ingest) * 100
        print(f"    * **{source}**: {pct:.1f}% ({row['total_size']} chars)")
        if pct > 5:
            gemini_summary.append(f"- {source}: {pct:.1f}%")
            
    # --- By Metric Name (Size) ---
    print("\n  **By Metric Name (Sorted by Size, Top 10):**")
    gemini_summary.append("\nTop 3 Metrics by Size:")
    for (metric, source), row in metric_stats.sort_values(by='total_size', ascending=False).head(10).iterrows():
        pct = (row['total_size'] / total_metric_ingest) * 100
        print(f"    * **{metric}** (Source: {source})")
        print(f"        - **Ingest**: {pct:.1f}% ({row['total_size']} chars)")
        if len(gemini_summary) < 10: # Limit summary
             gemini_summary.append(f"- {metric} ({source}): {pct:.1f}%")

    # --- By Metric Group (Size) ---
    print("\n  **By Metric Group (Sorted by Size, Top 10):**")
    gemini_summary.append("\nTop 3 Metric Groups by Size:")
    for (group, source), row in group_stats.sort_values(by='total_size', ascending=False).head(10).iterrows():
        pct = (row['total_size'] / total_metric_ingest) * 100
        print(f"    * **{group}** (Source: {source})")
        print(f"        - **Ingest**: {pct:.1f}% ({row['total_size']} chars)")
        if len(gemini_summary) < 15: # Limit summary
             gemini_summary.append(f"- {group} ({source}): {pct:.1f}%")
             
    # === SECTION 2: BY LOG COUNT (FREQUENCY) ===
    print("\n" + ("=" * 20))
    print("  ### Breakdown by Log Count (Frequency Analysis) ###")
    print("=" * 20)

    # --- By Source (Count) ---
    print("\n  **By Source (Sorted by Count):**")
    for source, row in source_stats.sort_values(by='total_count', ascending=False).head(5).iterrows():
        pct = (row['total_count'] / total_metric_count) * 100
        print(f"    * **{source}**: {pct:.1f}% ({row['total_count']} logs)")
            
    # --- By Metric Name (Count) ---
    print("\n  **By Metric Name (Sorted by Count, Top 10):**")
    for (metric, source), row in metric_stats.sort_values(by='total_count', ascending=False).head(10).iterrows():
        pct = (row['total_count'] / total_metric_count) * 100
        print(f"    * **{metric}** (Source: {source})")
        print(f"        - **Count**: {pct:.1f}% ({row['total_count']} logs)")

    # --- By Metric Group (Count) ---
    print("\n  **By Metric Group (Sorted by Count, Top 10):**")
    for (group, source), row in group_stats.sort_values(by='total_count', ascending=False).head(10).iterrows():
        pct = (row['total_count'] / total_metric_count) * 100
        print(f"    * **{group}** (Source: {source})")
        print(f"        - **Count**: {pct:.1f}% ({row['total_count']} logs)")

    print(f"\n  ...MetricName deep dive complete.")
    return "\n".join(gemini_summary)
# --- END MODIFIED FUNCTION ---


def generate_insights_summary(df, total_logs, sorted_stats, metric_summary=None):
    """
    Generates a concise text summary of the statistical analysis
    to be used as context for the Gemini API call.
    """
    print("\n" + ("=" * 20))
    print("  Generating statistical summary as input...")
    
    summary = []
    summary.append(f"Log Sample Analysis (Total Logs: {total_logs})\n")
    
    summary.append("--- Top 5 Most Present Attributes ---")
    for item in sorted_stats[:5]:
        summary.append(f"- {item['attribute']} (Presence: {item['presence_pct']:.1f}%, "
                       f"Unique Values: {item['unique_values']})")

    summary.append("\n--- Top 5 Largest Attributes (by 90th Percentile) ---")
    # Sort stats by 90th percentile length
    sorted_by_size = sorted(sorted_stats, key=lambda x: -x['p90_length'])
    for item in sorted_by_size[:5]:
        if item['p90_length'] > 0:
            summary.append(f"- {item['attribute']} (P90 Size: {item['p90_length']:.0f} chars, "
                           f"Contribution: {item['contribution_pct']:.1f}%)")
    
    summary.append("\n--- Top 5 Most Frequent Log Messages ---")
    message_col = find_first_column(df.columns, ['message'])
    if message_col:
        top_messages = df[message_col].astype(str).str.strip().value_counts().head(5)
        for msg, count in top_messages.items():
            msg_short = msg[:100] + "..." if len(msg) > 100 else msg
            summary.append(f"- (Count: {count}) \"{msg_short}\"")
    
    summary.append("\n--- Key Attribute Examples (for infrastructure detection) ---")
    key_attrs = ['filepath', 'filePath', 'host', 'hostname', 'cluster_name',
                 'container_name', 'platform', 'entity.name', 'logger', 'logtype']
    for attr in key_attrs:
        # Check for normalized AND deduplicated names
        present_cols = [c for c in df.columns if c.startswith(attr.lower())]
        for col in present_cols:
            examples = df[col].dropna().unique()[:3]
            if len(examples) > 0:
                 summary.append(f"- Examples for '{col}': {', '.join([str(e) for e in examples])}")
                 
    summary.append("\n--- Security/Proxy Attribute Examples ---")
    proxy_attrs = ['xffheaderoriginalvalues', 'clientip', 'errorcontent']
    for attr in proxy_attrs:
         present_cols = [c for c in df.columns if c.startswith(attr.lower())]
         for col in present_cols:
            examples = df[col].dropna().unique()[:2]
            if len(examples) > 0:
                 summary.append(f"- Examples for '{col}': {', '.join([str(e) for e in examples])}")

    # --- NEW: Add Metric Summary if it exists ---
    if metric_summary:
        summary.append("\n--- MetricName Deep Dive Summary ---")
        summary.append(metric_summary)
    # --- END NEW ---

    return "\n".join(summary)


def call_gemini_for_insights(summary_text, api_key):
    """
    Calls the Gemini API with the statistical summary to get
    a natural language analysis. (Now synchronous using 'requests')
    """
    print("\n" + ("=" * 20))
    print("  Running advanced analysis...")
    print("  (This may take a few seconds)...")
    
    # We must use the 'gemini-2.5-flash-preview-09-2025' model
    model = "gemini-2.5-flash-preview-09-2025"
    # model = "gemini-2.5-pro"
    
    # --- THIS IS THE CORRECTED LINE ---
    # The URL for the generateContent endpoint (using Python f-string)
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    # --- END CORRECTION ---
    
    system_prompt = (
        "You are an expert-level Site Reliability Engineer (SRE) and FinOps (Cloud Cost) analyst. "
        "Your job is to analyze a statistical summary from a log sample and provide a natural language summary. "
        "First, describe the *infrastructure* and *application stack* you can infer from the attributes. "
        "Second, identify specific, actionable anomalies related to *cost*, *performance*, or *security*. "
        "Cite the log evidence (e.g., messages, attributes) from the summary in your analysis. "
        "Format your response in clean markdown."
    )
    
    user_prompt = (
        "Here is the statistical summary from a log file analysis. Please provide your "
        "SRE/FinOps analysis as described in your system instructions.\n\n"
        "--- ANALYSIS SUMMARY --- \n"
        f"{summary_text}"
    )

    payload = {
        "contents": [{ "parts": [{ "text": user_prompt }] }],
        "systemInstruction": {
            "parts": [{ "text": system_prompt }]
        }
    }
    
    headers = { "Content-Type": "application/json" }

    try:
        response = requests.post(api_url, json=payload, headers=headers)

        if not response.ok:
            print(f"  ...Gemini API Error: {response.status_code} {response.reason}", response.text)
            return f"Error: Gemini API call failed with status {response.status_code}."

        result = response.json()
        candidate = result.get('candidates', [{}])[0]
        text = candidate.get('content', {}).get('parts', [{}])[0].get('text')

        if text:
            print("  ...Gemini analysis complete.")
            return text
        else:
            print("  ...Gemini API Error: No valid text response found in candidate.", result)
            return "Error: Received an invalid or empty response from the API."
            
    except requests.exceptions.RequestException as e:
        print(f"  ...An error occurred during the API request: {e}")
        return f"Error: Failed to call API. {e}"
    except Exception as e:
        print(f"  ...An unexpected error occurred during Gemini call: {e}")
        return f"Error: {e}"


# --- Main Execution ---

def main():
    """
    Main function to parse arguments and run the analyses.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Analyzes a JSON or CSV log sample file.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        prog='attribute-analyzer.py'  # Set the program name
    )
    # --- Arguments ---
    parser.add_argument(
        'filepath',
        metavar='<your_log_file.json_or_csv>',
        type=str,
        help='The path to the JSON or CSV log file to analyze.'
    )
    
    # --- Tunable Arguments (All existing args remain) ---
    parser.add_argument(
        '--PRESENCE_THRESHOLD_PCT',
        type=float,
        default=25.0,
        help='Threshold for an attribute to be "high presence". Default: 25.0'
    )
    parser.add_argument(
        '--LARGE_ATTR_CHAR_LENGTH',
        type=int,
        default=50,
        help='Character length to define a "large attribute". Default: 50'
    )
    parser.add_argument(
        '--LARGE_ATTR_PERCENTILE',
        type=float,
        default=0.5,
        help='Percentile to check for large attributes (e.g., 0.5 = 50th percentile). Default: 0.5'
    )
    parser.add_argument(
        '--LARGE_ATTR_PRESENCE_THRESHOLD',
        type=float,
        default=0.2,
        help='Presence threshold for large attribute analysis (e.g., 0.2 = 20%%). Default: 0.2'
    )
    parser.add_argument(
        '--LOG_HASH_FREQUENCY_THRESHOLD',
        type=float,
        default=0.015,
        help='Report duplicate log hashes that exceed this frequency (e.g., 0.015 = 1.5%%). Default: 0.015'
    )
    parser.add_argument(
        '--PAYLOAD_SIZE_PERCENTILE',
        type=float,
        default=0.99,
        help='Report on logs in the top percentile of payload size (e.g., 0.99 = top 1%%). Default: 0.99'
    )
    parser.add_argument(
        '--PAYLOAD_SIZE_HASH_FREQUENCY',
        type=float,
        default=0.01,
        help='For large payloads, report hashes that exceed this frequency (e.g., 0.01 = 1%%). Default: 0.01'
    )
    
    # --- GEMINI FLAGS ---
    parser.add_argument(
        '--analyze_with_gemini',
        action='store_true',
        help='Send a statistical summary to the Gemini API for advanced analysis.'
    )
    parser.add_argument(
        '--GEMINI_API_KEY',
        type=str,
        default=None,
        help='Your Gemini API key. Required if --analyze_with_gemini is used.'
    )

    args = parser.parse_args()
    
    print("\n--- Step 1/5: Loading Log File ---")
    start_load = time.time()
    df = load_log_file(args.filepath)
    if df is None or df.empty:
        print("\nExiting due to file loading error or empty file.")
        return
    end_load = time.time()
    print(f"--- File loaded in {end_load - start_load:.2f}s ---")
    

    print("\n--- Step 2/5: Analyzing Attributes ---")
    total_logs, sorted_stats = analyze_attributes(df)
    
    print("\n--- Step 3/5: Generating Summary Reports ---")
    start_report = time.time() 
    best_attributes = print_best_attributes(total_logs, sorted_stats, args.PRESENCE_THRESHOLD_PCT)
    print_combination_analysis(best_attributes)
    end_report = time.time()
    # --- THIS IS THE CORRECTED LINE ---
    print(f"--- Reports generated in {end_report - start_report:.2f}s ---")
    # --- END CORRECTION ---
    
    # --- NEW: Step 4: Calculate Hashes and Sizes ---
    # This is moved here so 'log_total_size' is available for all anomaly steps
    print("\n--- Step 4/5: Calculating Hashes and Payload Sizes ---")
    size_threshold, df_with_metrics = calculate_log_hashes_and_size(df, args.PAYLOAD_SIZE_PERCENTILE)
    # We use df_with_metrics (which contains log_hash, log_total_size) for all subsequent analysis
    # If hashing failed (e.g., no 'message'), df_with_metrics might be None or just lack 'log_hash'
    if df_with_metrics is None:
        print("...Hashing/size calculation failed, falling back to original DataFrame for remaining analysis.")
        df_with_metrics = df.copy() # Use a copy to avoid side-effects
        if 'log_total_size' not in df_with_metrics.columns:
             print("...Manually calculating log_total_size as fallback.")
             df_with_metrics['log_total_size'] = df.astype(str).apply(lambda x: x.str.len()).sum(axis=1)


    # --- MODIFIED: Step 5: Anomaly & Metric Analysis ---
    print("\n--- Step 5/5: Analyzing Anomalies & Metrics ---")
    
    # --- THIS IS THE CORRECTED LINE ---
    # Run the standard anomaly insights
    print_all_anomaly_insights(df_with_metrics, size_threshold, total_logs, TOP_ANOMALOUS_MESSAGES,
                               args.LOG_HASH_FREQUENCY_THRESHOLD,
                               args.PAYLOAD_SIZE_HASH_FREQUENCY,
                               args.LARGE_ATTR_CHAR_LENGTH,
                               args.LARGE_ATTR_PERCENTILE,
                               args.LARGE_ATTR_PRESENCE_THRESHOLD)
    # --- END CORRECTION ---

    # --- NEW: Run the MetricName Deep Dive ---
    # This function will print its own header
    metric_summary = print_metricname_deep_dive(df_with_metrics, total_logs)
    

    # --- NEW: Step 6 (Conditional) ---
    if args.analyze_with_gemini:
        # --- Check for API Key ---
        if not args.GEMINI_API_KEY:
            print("\n" + ("!" * 60))
            print("### ERROR: Missing Gemini API Key ###")
            print("To use --analyze_with_gemini, you must also provide your API key:")
            print("  --GEMINI_API_KEY \"YOUR_API_KEY_HERE\"")
            print("You can generate a key at https://aistudio.google.com/app/apikey")
            print(("!" * 60) + "\n")
        else:
            print("\n--- Step 6/6: Generating Advanced Analysis with Gemini ---")
            # 1. Generate the summary, now including the metric_summary
            summary_text = generate_insights_summary(df, total_logs, sorted_stats, metric_summary)
            
            # 2. Call the API
            gemini_response = call_gemini_for_insights(summary_text, args.GEMINI_API_KEY)
            
            # 3. Print the response
            print_header("Advanced Analysis")
            print(gemini_response)


    print("\n" + ("-" * 60))
    print("Analysis Complete.")
    print(("-" * 60) + "\n")


if __name__ == "__main__":
    main()