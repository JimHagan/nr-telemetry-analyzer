# Telemetry Ingest & Attribute Analyzer

This Python script, `nr-telemetry-analyzer.py`, loads and analyzes a JSON or CSV log sample file to provide deep insights into log ingest patterns. It is designed to help you understand *what* you are logging, *where* it's coming from, and *how* to optimize your ingest for cost and performance.

The script reports on:
1.  **Attribute Popularity & Size:** Which attributes are most common and how much data they contribute.
2.  **Facet Recommendations:** A recommended NRQL query for subdividing your data.
3.  **Actionable Anomalies:** A powerful 5-part analysis to find the specific logs that are driving the most cost, including:
    * Functionally identical (duplicate) logs.
    * Logs that are both *large* and *frequent*.
    * Repetitive, "chatty" log messages.
    * Attributes that are storing massive 
    * Broken or truncated multi-line logs.

## Dependencies

The script requires the following:
* **Python 3.7+**
* **pandas**: Used for all core data analysis.
* **numpy**: A dependency of pandas, used for `NaN` checking.
* **requests**

## Setup and Installation

It is highly recommended to run this script within a Python virtual environment to manage dependencies.

### 1. Create a Virtual Environment

From your terminal, navigate to the directory where you saved `nr-telemetry-analyzer.py` and create a virtual environment:

```sh
# For macOS and Linux
python3 -m venv venv

# For Windows
python -m venv venv
````

### 2\. Activate the Virtual Environment

You must activate the environment in your terminal session before installing dependencies or running the script.

```sh
# For macOS and Linux
source venv/bin/activate

# For Windows (Command Prompt)
.\venv\Scripts\activate.bat

# For Windows (PowerShell)
.\venv\Scripts\Activate.ps1
```

Your terminal prompt should change to show `(venv)` at the beginning.

### 3\. Install Dependencies

With your virtual environment active, install the required libraries:

```sh
pip install pandas numpy requests
```

## How to Run

Run the script from your terminal by passing the path to your JSON or CSV log sample file as the main argument.

### Basic Usage

```sh
python nr-telemetry-analyzer.py "path/to/your-log-sample.csv"
```

### Advanced Usage with Overrides

You can override the default analysis thresholds using command-line arguments.

```sh
# Run with a stricter presence threshold for attribute analysis
python nr-telemetry-analyzer.py "sample.json" --PRESENCE_THRESHOLD_PCT 50

# Run with more aggressive "large payload" detection
python nr-telemetry-analyzer.py "sample.csv" --PAYLOAD_SIZE_PERCENTILE 0.95 --LARGE_ATTR_CHAR_LENGTH 250
```

### All Command-Line Arguments

  * `filepath`: (Required) The path to the JSON or CSV log file to analyze.

  * `--PRESENCE_THRESHOLD_PCT`:

      * Sets the "high presence" filter for the "Attribute Analysis" section.
      * Default: `25.0` (25%)

  * `--LARGE_ATTR_CHAR_LENGTH`:

      * Sets the minimum character length to consider an attribute "large".
      * Default: `50`

  * `--LARGE_ATTR_PERCENTILE`:

      * Sets the percentile to check for the "Large Attribute" analysis.
      * Default: `0.5` (50th percentile, or median)

  * `--LARGE_ATTR_PRESENCE_THRESHOLD`:

      * An attribute must be present on this percentage of logs to be included in the "Large Attribute" analysis.
      * Default: `0.2` (20%)

  * `--LOG_HASH_FREQUENCY_THRESHOLD`:

      * Reports duplicate log hashes that appear *more* frequently than this percentage.
      * Default: `0.015` (1.5%)

  * `--PAYLOAD_SIZE_PERCENTILE`:

      * Defines the percentile for "large" payloads (e.g., 0.99 finds logs in the top 1%).
      * Default: `0.99` (99th percentile)

  * `--PAYLOAD_SIZE_HASH_FREQUENCY`:

      * For logs that meet the "large payload" percentile, this sets the minimum frequency to be reported.
      * Default: `0.01` (1%)

-----

## Interpreting the Output

The script prints its analysis directly to the terminal in four or five steps.

### Step 1: Log Sample Count

This is the total number of log entries (JSON objects or CSV rows) found in the provided sample file.

### Step 2: Attribute Analysis for Ingest Subdivision

This section lists the best attributes for "faceting" or "segmenting" your log data, sorted by their **total size contribution**.

**Example Output:**

NOTE: Output has markdown formatting

```
**message**
    * **Total Size Contribution:** 42.51%
    * **Presence:** 100.0% (25000 out of 25000 logs)
    * **Unique Values:** 7210
    * **Max Length:** 1024 chars
    * **50th Percentile Length:** 120 chars
    * **90th Percentile Length:** 350 chars
    * **Examples:** "Status requested.", "failed to publish events"
```

  * **Total Size Contribution:** The most important stat. This shows what percentage of the *total data size* in the sample this one attribute is responsible for. `message` is often the largest.
  * **Presence:** How often this attribute appears.
  * **Unique Values:** How many different values it has. Low numbers (like `level`) are good for grouping.
  * **Percentile Length:** Shows the 50th (median) and 90th percentile length, helping you understand the *typical* size vs. the *outlier* size.

### Step 3: Attribute Combination Analysis

This section takes the top attributes from Step 2 (selected by *presence* and *cardinality*) and provides a sample NRQL query. This is a great starting point for building a dashboard to visualize your log sources.

### Step 4: Potential Anomaly Insights

This is the most powerful section. It runs 5 different analyses to find specific, actionable insights.

**1. Duplicate Log Hash Analysis**

  * **What it does:** Finds *functionally identical* logs. It hashes every log after *removing* unique IDs, timestamps, and pod/host names.
  * **What it finds:** Logs that are identical in every way (`message`, `level`, `container_name`, etc.) and are being generated from many different sources. This is a key indicator of "chatter."

**2. Large Payload Hash Analysis**

  * **What it does:** Finds the "worst offenders." It first identifies all logs in the top percentile of payload size (e.g., top 1%) and then checks if any of *those* logs are also highly frequent.
  * **What it finds:** Logs that are both **very large** and **very frequent**. These are primary targets for cost optimization, as they contribute disproportionately to ingest.

**3. High-Frequency Message Analysis**

  * **What it does:** This is similar to the duplicate hash, but it only groups by `message` and high-level context (like `container_name`, `level`, `environment`). It intentionally *ignores* pod names and hosts.
  * **What it finds:** Repetitive messages coming from *many different pods* of the same service. This is why you might see two "started call" anomalies that look identical except for the `pod_name`—it's flagging that the *same message* is a problem on *multiple pods*.

**4. Large Attribute Analysis**

  * **What it does:** This checks each *individual attribute* (like `error.stack`, `payload.body`, etc.) to see if it's consistently large.
  * **What it finds:** Attributes that are being used to store large JSON payloads or full stack traces. This can dramatically increase log size.

**5. Truncated Log Analysis**

  * **What it does:** This performs a simple check: "Does the `message` field end with a newline character (`\n`)?"
  * **What it finds:** Broken multi-line logs. This is a classic sign that a stack trace has been split into 10-20 separate log entries, inflating log counts and making debugging impossible. This can almost always be fixed in your log forwarder configuration.
