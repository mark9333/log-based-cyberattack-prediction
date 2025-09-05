import re
import matplotlib.pyplot as plt
from collections import defaultdict

LINE_FORMAT = re.compile(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} - \[[A-Za-z]+\]\s+.+$')

PATTERNS = {
    "unauthorized_access": r"suspicious activity detected: unauthorized_access",
    "malware": r"suspicious activity detected: malware",
    "ddos": r"suspicious activity detected: ddos",
    "sql_injection": r"suspicious activity detected: sql_injection",
    "ip_blocked": r"suspicious activity detected: ip_blocked",
    "brute_force": r"suspicious activity detected: brute_force",
    "intrusion": r"suspicious activity detected: intrusion",
    "port_scan": r"suspicious activity detected: port_scan"
}

def analyze_log(file_path):
    results = defaultdict(int)
    total_lines = 0
    invalid_lines = 0
    invalid_list = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, raw in enumerate(f, start=1):
            total_lines += 1
            line = raw.rstrip('\n')
            if not LINE_FORMAT.match(line.strip()):
                invalid_lines += 1
                invalid_list.append((idx, line))
                continue

            for key, pattern in PATTERNS.items():
                if re.search(pattern, line, re.IGNORECASE):
                    results[key] += 1

    return dict(results), total_lines, invalid_lines, invalid_list

def save_report(results, total_lines, file_path):
    report_path = file_path.replace(".log", "_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Total lines: {total_lines}\n")
        f.write("Suspicious activities detected:\n")
        for k, v in results.items():
            f.write(f"- {k}: {v} occurrence(s)\n")
    return report_path

def create_activity_chart(results, file_path):
    if not results:
        return

    keys = list(results.keys())
    values = list(results.values())

    plt.figure(figsize=(10, 5))
    plt.bar(keys, values, color='orange')
    plt.title("Suspicious Activity Detected")
    plt.xlabel("Activity Type")
    plt.ylabel("Occurrences")
    plt.yticks(range(0, max(values) + 2, 5))
    plt.tight_layout()

    chart_path = file_path.replace(".log", "_chart.png")
    plt.savefig(chart_path)
    plt.close()