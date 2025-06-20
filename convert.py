import re
from datetime import datetime
from typing import Dict, Optional

def parse_bsub_log(text: str) -> Dict[str, Optional[str]]:
    """Parse a BSUB log file and extract relevant job information."""
    # Initialize fields with default values
    fields = {
        "Job ID": None,
        "Category": None,
        "Status": "Unknown",
        "Fail Reason": "",
        "Start Time": None,
        "Finish Time": None,
        "CPU Cores": None,
        "GPU": None,
        "Epochs": None,
        "Batch sizes": None,
        "Loss types": None,
        "Encoders": None,
        "Freeze": None,
        "Optimizers": None,
        "Learning rates": None,
        "Resize": None,
        "Outer folds": None,
        "Inner folds": None,
        "Combinations": "=Jobs[Inner folds]*Jobs[Outer folds]*COUNTA(SPLIT(Jobs[Batch sizes], \", \", \"\", TRUE))*COUNTA(SPLIT(Jobs[Loss types], \", \", \"\", TRUE))*COUNTA(SPLIT(Jobs[Encoders], \", \", \"\", TRUE))*COUNTA(SPLIT(Jobs[Freeze], \", \", \"\", TRUE))*COUNTA(SPLIT(Jobs[Optimizers], \", \", \"\", TRUE))*COUNTA(SPLIT(Jobs[Learning rates], \", \", \"\", TRUE))*COUNTA(SPLIT(Jobs[Resize], \", \", \"\", TRUE))+Jobs[Outer folds]",
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

    # Extract CPU cores
    if match := re.search(r"#BSUB\s+-n\s+(\d+)", text):
        fields["CPU Cores"] = match.group(1)

    # Extract GPU from queue in "Submitted from host ... to Queue <gpua100>"
    if match := re.search(r"Submitted from host.+to Queue <gpu(\w+)>", text):
        fields["GPU"] = match.group(1)

    # Determine status and fail reason
    if "Done successfully" in text or "Completed <exit>" in text:
        fields["Status"] = "Done"
    elif "Started on" in text:
        fields["Status"] = "Running"
    elif "PENDING TIME DETAILS" in text:
        fields["Status"] = "Pending"
    elif "Exited" in text:
        fields["Status"] = "Failed"
        if "TERM_OWNER" in text:
            fields["Status"] = "Cancelled"
            fields["Fail Reason"] = "Killed by user/admin"

    # Extract and format start time (from "Dispatched" line for consistency with expected output)
    if match := re.search(r"([A-Z][a-z]{2}\s+\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\d{4}).*Dispatched", text):
        try:
            dt = datetime.strptime(match.group(1), "%a %b %d %H:%M:%S %Y")
            fields["Start Time"] = dt.strftime("%m/%d/%Y %H:%M:%S")
        except ValueError:
            pass

    # Extract and format finish time
    if match := re.search(r"([A-Z][a-z]{2}\s+\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\d{4}).*Done successfully", text):
        try:
            dt = datetime.strptime(match.group(1), "%a %b %d %H:%M:%S %Y")
            fields["Finish Time"] = dt.strftime("%m/%d/%Y %H:%M:%S")
        except ValueError:
            pass


    def extract_param(flag: str):
        """Extract parameter values following a flag, handling quoted and unquoted values."""
        if match := re.search(rf'--{flag}\s*((?:"[^"]*"\s*|\d+\.\d+\s*|\d+\s*)+)(?:\\;|$)', text):
            value: str = match.group(1)
            value = re.sub(r"\s{2,}", "", value).replace(r"\n", " ")
            value = re.sub(r'""', '" "', value)
            value = re.sub(r"''", "' '", value)
            if match := re.search(pattern:=r'(--\w*)"', value):
                value = re.sub(pattern, rf'{match.group(1)} "', value)
            if match := re.search(pattern:=r'(--\w*)([0123456789])', value):
                value = re.sub(pattern, rf'{match.group(1)} {match.group(2)}', value)

            value = value.replace('"', '').replace("'", "").replace(",", "").strip()
            values = ", ".join(value.split(" "))
            # print(values)
            return values

        if flag=="outer" and "nsplits" in text:
            return extract_param("nsplits")

        return None

    # Map flags to field names and specify which are lists
    param_flags = {
        "epochs": "Epochs",
        "batchsizes": "Batch sizes",
        "loss": "Loss types",
        "encoders": "Encoders",
        "freeze": "Freeze",
        "optimizers": "Optimizers",
        "lr": "Learning rates",
        "resize": "Resize",
        "outer": "Outer folds",
        "inner": "Inner folds",
    }

    # Extract all parameters
    for flag, field in param_flags.items():
        fields[field] = extract_param(flag)

    return fields

if __name__ == "__main__":
    import sys
    if not len(sys.argv) > 1:
        print("Please provide a BSUB log file as an argument.")
    with open(sys.argv[1], "r") as f:
        log_text = f.read()
    parsed = parse_bsub_log(log_text)

    # print(parsed.keys())

    sheet = ["Status", "Job ID", "GPU", "Category", "Batch sizes", "Loss types", "Encoders", "Freeze", "Optimizers", "Learning rates", "Resize", "Outer folds", "Inner folds", "Combinations", "Epochs", "CPU Cores", "Fail Reason", "Start Time", "Finish Time", "Description"]

    print("\t".join([f"{parsed[k]}" for k in sheet if parsed[k] is not None]))
