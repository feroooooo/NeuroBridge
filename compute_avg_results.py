import os
import argparse

import pandas as pd

# Get input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', default="", type=str)

args = parser.parse_args()

if not os.path.exists(args.result_dir):
    raise Exception("wrong dir")

df_list = []
for run in sorted(os.listdir(args.result_dir)):
    if os.path.isdir(os.path.join(args.result_dir, run)):
        file = os.path.join(args.result_dir, run, "result.csv")
        df = pd.read_csv(file)
        df['sub'] = run[-6:]
        cols = ['sub'] + [col for col in df.columns if col != 'sub']
        df = df[cols]
        df_list.append(df)

# Concatenate all DataFrames
all_data = pd.concat(df_list, ignore_index=True)

# Extract numeric columns (excluding 'sub' and 'best epoch')
numeric_cols = all_data.columns.difference(['sub', 'best epoch'])

# Convert to float, keep two decimal places and pad with zeros (convert to string)
for col in numeric_cols:
    all_data[col] = all_data[col].astype(float).map(lambda x: f"{x:.1f}")

# Calculate average values (still using float for calculation, then formatting)
avg_values = all_data[numeric_cols].astype(float).mean()
avg_row = {col: f"{avg_values[col]:.1f}" for col in numeric_cols}
avg_row['sub'] = 'Average'

# Add average row
all_data = pd.concat([all_data, pd.DataFrame([avg_row])], ignore_index=True)

# Save the merged result
all_data.to_csv(os.path.join(args.result_dir, 'avg_results.csv'), index=False)

print(all_data)