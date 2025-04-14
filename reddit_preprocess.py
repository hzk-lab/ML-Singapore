import pandas as pd
import glob

# Map each filename to a subreddit label (in case you need it later)
file_map = {
    "nus_comments.csv": "NUS",
    "NationalServiceSG_comments.csv": "NationalServiceSG",
    "SGExams_comments.csv": "SGExams",
    "singapore_comments.csv": "Singapore"
}

# Read and process each CSV
dfs = []
for filename, subreddit in file_map.items():
    df = pd.read_csv(filename)
    df["source_subreddit"] = subreddit  # Tag each row with its subreddit
    dfs.append(df)

# Combine them
combined_df = pd.concat(dfs, ignore_index=True)

# Optional: Create a new column with combined post + comment text
combined_df["text"] = combined_df["post_title"].fillna("") + " " + combined_df["comment_body"].fillna("")
# Keep only what you need
final_df = combined_df[["text", "source_subreddit"]]
