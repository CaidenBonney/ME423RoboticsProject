#!/usr/bin/env python3
# src/test_data/toss2.txt
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# =========================
# EDIT THESE PATHS ONLY
# =========================
INPUT_FILE = Path("src/test_data/toss2.txt")
OUTPUT_DIR = Path("src/test_data/toss2_analysis")


BALL_RE = re.compile(r"BALL POSITION:\s*\[([^\]]+)\]")
ISSUED_PHI_RE = re.compile(r"PHI CMD:\s*\[([^\]]+)\]")
COMMANDED_PHI_RE = re.compile(r"commanded phi:\s*\[([^\]]+)\]")
COMMANDED_POS_RE = re.compile(r"commanded position:\s*\[([^\]]+)\]")
XYZ_POS_RE = re.compile(r"POSITION WITH OFFSET:\s*\[([^\]]+)\]")


def parse_vector(text):
    return [float(x) for x in text.replace(",", " ").split()]


def carry_forward_zero_ball(samples):
    last_valid = None
    cleaned = []

    for vec in samples:
        if vec == [0.0, 0.0, 0.0] and last_valid is not None:
            cleaned.append(last_valid.copy())
        else:
            cleaned.append(vec)
            if vec != [0.0, 0.0, 0.0]:
                last_valid = vec.copy()

    return cleaned


def parse_log(path):
    text = path.read_text(errors="ignore")
    lines = text.splitlines()

    latest_commanded_phi = None
    records = []

    for line in lines:
        commanded_phi_match = COMMANDED_PHI_RE.search(line)
        if commanded_phi_match:
            latest_commanded_phi = parse_vector(commanded_phi_match.group(1))

        ball_match = BALL_RE.search(line)
        if not ball_match:
            continue

        ball = parse_vector(ball_match.group(1))

        issued_phi_match = ISSUED_PHI_RE.search(line)
        issued_phi = parse_vector(issued_phi_match.group(1)) if issued_phi_match else None

        commanded_pos_match = COMMANDED_POS_RE.search(line)
        commanded_pos = parse_vector(commanded_pos_match.group(1)) if commanded_pos_match else None

        xyz_pos_match = XYZ_POS_RE.search(line)
        xyz_pos = parse_vector(xyz_pos_match.group(1)) if xyz_pos_match else None

        records.append(
            {
                "ball": ball,
                "issued_phi": issued_phi,
                "commanded_phi": latest_commanded_phi.copy() if latest_commanded_phi else None,
                "commanded_pos": commanded_pos,
                "xyz_pos": xyz_pos,
            }
        )

    if not records:
        raise ValueError("No BALL POSITION entries found in the log file.")

    cleaned_ball = carry_forward_zero_ball([r["ball"] for r in records])
    for record, cleaned in zip(records, cleaned_ball):
        record["ball"] = cleaned

    valid_records = [
        r for r in records
        if r["issued_phi"] is not None
        and r["commanded_phi"] is not None
        and r["commanded_pos"] is not None
        and r["xyz_pos"] is not None
    ]

    if not valid_records:
        raise ValueError(
            "No complete records found with BALL POSITION, PHI CMD, commanded phi, commanded position, and POSITION WITH OFFSET."
        )

    rows = []
    for idx, rec in enumerate(valid_records):
        ball = rec["ball"]
        issued_phi = rec["issued_phi"]
        commanded_phi = rec["commanded_phi"]
        commanded_pos = rec["commanded_pos"]
        xyz_pos = rec["xyz_pos"]

        rows.append(
            {
                "sample": idx,
                "time_s": idx / 30.0,  # 30 samples per second
                "ball_x": ball[0],
                "ball_y": ball[1],
                "ball_z": ball[2],
                "issued_phi_1": issued_phi[0] if len(issued_phi) > 0 else None,
                "issued_phi_2": issued_phi[1] if len(issued_phi) > 1 else None,
                "issued_phi_3": issued_phi[2] if len(issued_phi) > 2 else None,
                "issued_phi_4": issued_phi[3] if len(issued_phi) > 3 else None,
                "commanded_phi_1": commanded_phi[0] if len(commanded_phi) > 0 else None,
                "commanded_phi_2": commanded_phi[1] if len(commanded_phi) > 1 else None,
                "commanded_phi_3": commanded_phi[2] if len(commanded_phi) > 2 else None,
                "commanded_phi_4": commanded_phi[3] if len(commanded_phi) > 3 else None,
                "commanded_x": commanded_pos[0] if len(commanded_pos) > 0 else None,
                "commanded_y": commanded_pos[1] if len(commanded_pos) > 1 else None,
                "commanded_z": commanded_pos[2] if len(commanded_pos) > 2 else None,
                "xyz_x": xyz_pos[0] if len(xyz_pos) > 0 else None,
                "xyz_y": xyz_pos[1] if len(xyz_pos) > 1 else None,
                "xyz_z": xyz_pos[2] if len(xyz_pos) > 2 else None,
            }
        )

    return pd.DataFrame(rows)


def save_plot(df, y_columns, labels, title, ylabel, output_path):
    plt.figure(figsize=(10, 5))
    for col, label in zip(y_columns, labels):
        plt.plot(df["time_s"], df[col], label=label, marker='+')

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = parse_log(INPUT_FILE)

    csv_path = OUTPUT_DIR / "extracted_data.csv"
    df.to_csv(csv_path, index=False)

    save_plot(
        df,
        ["ball_y", "commanded_phi_1", "issued_phi_1"],
        ["Ball Y Position", "Commanded Phi[0]", "Issued Phi[0]"],
        "Ball Y Position and First-Angle Commanded vs Issued Values",
        "Value",
        OUTPUT_DIR / "ball_y_and_phi1.png",
    )

    save_plot(
        df,
        ["ball_x"],
        ["Ball X Position"],
        "Ball X Position",
        "Position",
        OUTPUT_DIR / "ball_x.png",
    )

    save_plot(
        df,
        ["ball_y"],
        ["Ball Y Position"],
        "Ball Y Position",
        "Position",
        OUTPUT_DIR / "ball_y.png",
    )

    save_plot(
        df,
        ["ball_z"],
        ["Ball Z Position"],
        "Ball Z Position",
        "Position",
        OUTPUT_DIR / "ball_z.png",
    )

    save_plot(
        df,
        ["ball_x", "commanded_x", "xyz_x"],
        ["Ball X Position", "Commanded X Position", "XYZ X Position"],
        "Overlay of Ball, Commanded, and XYZ X Position",
        "Position",
        OUTPUT_DIR / "overlay_x.png",
    )

    save_plot(
        df,
        ["ball_y", "commanded_y", "xyz_y"],
        ["Ball Y Position", "Commanded Y Position", "XYZ Y Position"],
        "Overlay of Ball, Commanded, and XYZ Y Position",
        "Position",
        OUTPUT_DIR / "overlay_y.png",
    )

    save_plot(
        df,
        ["ball_z", "commanded_z", "xyz_z"],
        ["Ball Z Position", "Commanded Z Position", "XYZ Z Position"],
        "Overlay of Ball, Commanded, and XYZ Z Position",
        "Position",
        OUTPUT_DIR / "overlay_z.png",
    )

    print("Done.")
    print(f"Samples parsed: {len(df)}")
    print(f"CSV saved to: {csv_path}")
    print(f"Plots saved to: {OUTPUT_DIR.resolve()}")


main()