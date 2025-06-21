import re
from typing import Dict, Optional
import pandas as pd
import subprocess
from datetime import datetime

def parse_bsub_log(text: str) -> Dict[str, Optional[str]]:
    """Parse a BSUB log file and extract relevant job information."""
    # Initialize fields with default values
    fields = {
        "Status": None,
        "Job ID": None,
        "GPU": None,
        "Category": None,
        "Batch sizes": None,
        "Loss types": None,
        "Encoders": None,
        "Freeze": None,
        "Optimizers": None,
        "Learning rates": None,
        "Resize": None,
        "Outer folds": None,
        "Inner folds": None,
        "Combinations": "",
        "Epochs": None,
        "CPU Cores": None,
        "Type": None,
        "Fail Reason": None,
        "Submitted Time": None,
        "Start Time": None,
        "Finish Time": None,
        "Elapsed Time": None,
        "Description": None,
    }

    # Normalize text: collapse multiple spaces, remove newlines and escaped semicolons
    text = re.sub(r"\s{2,}", "", log_text).strip()

    # Extract job ID
    if match := re.search(r"Job\s+<(\d+)>", text):
        fields["Job ID"] = match.group(1)

    # Extract category (job name)
    if match := re.search(r"Job Name\s+<([^>]+)>", text):
        fields["Category"] = match.group(1)

    # Extract GPU from queue in "Submitted from host ... to Queue <gpua100>"
    if match := re.search(r"Submitted from host.+to Queue <gpu(\w+)>", text):
        fields["GPU"] = match.group(1)

    if match := re.search(r'#BSUB -gpu "num=\d+:mode=(\w+)";', text):
        fields["Type"] = match.group(1)

    # Determine status and fail reason
    if "job killed by owner" in text.lower():
        fields["Status"] = "Cancelled"
        fields["Fail Reason"] = "Killed by user/admin"
    elif "Done successfully." in text:
        fields["Status"] = "Done"
    elif "Exited with exit code" in text:
        fields["Status"] = "Failed"

    time_pattern = r"\w{3} (\w{3}) (\d{1,2}) (\d{2}):(\d{2}):(\d{2}) (\d{4})"
    # Month, Day, Hour, Minute, Second, Year

    month_idx = {
        "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
        "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
        "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"
    }

    # Extract and format submitted time (from "Submitted from host" line for consistency with expected output)
    if match := re.search(rf"{time_pattern}:.*Submitted from host", text):
        try:
            time = f"{month_idx[match.group(1)]}/{match.group(2)}/{match.group(6)} {match.group(3)}:{match.group(4)}:{match.group(5)}"
            fields["Submitted Time"] = time
        except ValueError:
            pass

    # Extract and format start time (from "Starting" line for consistency with expected output)
    if match := re.search(rf"{time_pattern}:.*Starting \(Pid ", text):
        try:
            time = f"{month_idx[match.group(1)]}/{match.group(2)}/{match.group(6)} {match.group(3)}:{match.group(4)}:{match.group(5)}"
            fields["Start Time"] = time
        except ValueError:
            pass

    # Extract and format finish time
    if match := re.search(rf"Summary of time in seconds spent in various states by.*{time_pattern}", text):
        try:
            # dt = datetime.strptime(match.group(1), "%a %b %d %H:%M:%S %Y")
            # fields["Finish Time"] = dt.strftime("%m/%d/%Y %H:%M:%S")
            time = f"{month_idx[match.group(1)]}/{match.group(2)}/{match.group(6)} {match.group(3)}:{match.group(4)}:{match.group(5)}"
            fields["Finish Time"] = time
        except ValueError:
            pass

    # Extract elapsed time
    try:
        time_format = "%m/%d/%Y %H:%M:%S"
        dt1 = datetime.strptime(fields["Start Time"], time_format)
        dt2 = datetime.strptime(fields["Finish Time"], time_format)

        # Calculate the difference
        time_diff = dt2 - dt1
        fields["Elapsed Time"] = datetime.strftime(time_diff, time_format)
    except TypeError:
        pass


    def extract_param(flag: str):
        """Extract parameter values following a flag, handling quoted and unquoted values."""
        if match := re.search(rf'--{flag}\s*((?:"[^"]*"\s*|\d+\.\d+\s*|\d+\s*)+)(?:\\;|>,)', text):
            value: str = match.group(1)
            value = re.sub(r"\s{2,}", "", value).replace(r"\n", " ")
            value = re.sub(r'""', '" "', value)
            value = re.sub(r"''", "' '", value)
            if match := re.search(pattern:=r'(--\w*)"', value):
                value = re.sub(pattern, rf'{match.group(1)} "', value)
            if match := re.search(pattern:=r'(--\w*)([0123456789])', value):
                value = re.sub(pattern, rf'{match.group(1)} {match.group(2)}', value)

            value = value.replace('"', '').replace("'", "").replace(",", "").strip()
            values = value.split(" ")

            # Get unique values and remove empty strings
            values = list(set(values))
            values = [v for v in values if v]  # Remove empty strings

            # /work3/s234843/02466-Project/trained_models/dino/fdnzv5c2/dino-fdnzv5c2-ep=399.ckpt
            if flag == "encoder" or flag == "encoders":
                values = [match.group(1) if (match := re.search(r".*(dino-.*)-.*\.ckpt", v)) else v for v in values]
            values = ", ".join(values)
            
            return values

        if flag=="outer" and "nsplits" in text:
            return extract_param("nsplits")
        
        if flag=="batchsize" and "batchsizes" in text:
            return extract_param("batchsizes")
        
        if flag=="encoder" and "encoders" in text:
            return extract_param("encoders")
        
        if flag=="optimizer" and "optimizers" in text:
            return extract_param("optimizers")

        return None

    # Map flags to field names and specify which are lists
    param_flags = {
        "epochs": "Epochs",
        "batchsize": "Batch sizes",
        "loss": "Loss types",
        "encoder": "Encoders",
        "freeze": "Freeze",
        "optimizer": "Optimizers",
        "lr": "Learning rates",
        "resize": "Resize",
        "outer": "Outer folds",
        "inner": "Inner folds",
        "workers": "CPU Cores"
    }

    # Extract all parameters
    for flag, field in param_flags.items():
        fields[field] = extract_param(flag)

    # Extract CPU cores
    if match := re.search(r"#BSUB\s+-n\s+(\d+)", text):
        if int(match.group(1)) != int(fields["CPU Cores"]):
            fields["CPU Cores"] = f"{fields["CPU Cores"]} ({match.group(1)})"

    return fields

def get_bhist_output(job_id: int) -> str:
    """Runs the `bhist -b <job_id>` command and returns its output as text."""
    try:
        result = subprocess.run(["bhist", "-b", str(job_id)], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error retrieving bhist output for job {job_id}: {e}")
        return ""

if __name__ == "__main__":
    from tqdm import tqdm
    import os
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("ids", help = "Input one or more bjob ids", nargs='+')
    args = parser.parse_args()

    all_rows = []
    try:
        for job_id in tqdm(args.ids, desc="Processing jobs", unit="job"):
            job_id = int(job_id)
            log_text = get_bhist_output(job_id)
            if "No matching job found" in log_text.strip():
                continue
            parsed = parse_bsub_log(log_text)
            all_rows.append(parsed)

    except ValueError:
        for id in args.ids:
            if isinstance(id, str) and os.path.exists(id):
                with open(id, 'r') as f:
                    log_text = f.read()
                    parsed = parse_bsub_log(log_text)
                    all_rows.append(parsed)
    
    except Exception as e:
        print(f"A {type(e)} occurred: {e}")
        exit(1)


    if not all_rows:
        print("No valid job data parsed. Exiting.")
        exit(1)

    # Define the column order for the Excel sheet
    sheet_columns = [
        "Status", "Job ID", "GPU", "Category", "Batch sizes", "Loss types", "Encoders",
        "Freeze", "Optimizers", "Learning rates", "Resize", "Outer folds", "Inner folds",
        "Combinations", "Epochs", "CPU Cores", "Type", "Fail Reason", "Submitted Time", "Start Time", "Finish Time", "Elapsed Time",
        "Description"
    ]
    
    df = pd.DataFrame(all_rows, columns=sheet_columns)

    output_file = "parsed_bsub_log"
    df.to_excel(output_file + ".xlsx", index=False, sheet_name="BSUB Log")
    df.to_csv(output_file + ".csv", index=False)

    print(f"Excel and CSV files saved as '{output_file}.xlsx' and '{output_file}.csv'.")