import os
import pandas as pd


__author__ = 'yuhao liu'

def merge_annotator_metrics():
    root_dir = "/home/yl241/experiments/fCLIPSeg/inter-annotator/csv"  # e.g., "./csv" in your working directory
    # 2. Collect all DataFrames
    df_list = []

    for annotator_folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, annotator_folder)
        if not os.path.isdir(folder_path):
            continue

        metrics_path = os.path.join(folder_path, "metrics.csv")
        if not os.path.isfile(metrics_path):
            continue

        # 3. Read each CSV
        df = pd.read_csv(metrics_path)

        # 4. Drop the 'step' column if it exists
        if "step" in df.columns:
            df = df.drop(columns=["step"])

        df_list.append(df)

    # 5. Concatenate all DataFrames
    if not df_list:
        raise RuntimeError("No metrics.csv files found under csv/annotator_*")
    combined = pd.concat(df_list, ignore_index=True)

    # 6. (Optional) Reorder columns so 'annotator' and 'img_id' come first
    cols = ["annotator", "img_id"] + [c for c in combined.columns if c not in ("annotator", "img_id")]
    combined = combined[cols]

    combined.to_csv(os.path.join(root_dir, "all_metrics.csv"), index=False)
    print("Merged CSV saved as all_metrics.csv")


if __name__ == "__main__":
    merge_annotator_metrics()