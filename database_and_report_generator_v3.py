# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 00:09:43 2025

@author: natha
"""

import os
from pathlib import Path
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import gc
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from io import BytesIO
from PIL import Image
from reportlab.lib.pagesizes import landscape, letter  # add this import if not already present
from datetime import datetime
from matplotlib.ticker import MultipleLocator
import matplotlib.patches as mpatches


####################################################
############## INPUTS ##############################
####################################################

####################################################
############## 870 Total Hero Power ################
####################################################
thp_flag = False # If TRUE, run analysis on total hero power
thp_metric = "Total_Hero_Power"
thp_server = 870
thp_clan_tag_to_filteron = "GODS"
thp_daterange_start = None
thp_daterange_end = None
thp_delta_daterange_start = "2025-10-11"
thp_delta_daterange_end = "2025-10-20"

####################################################
############## Enemy Total Hero Power ##############
####################################################
enemy_thp_flag = True # If TRUE, run analysis on enemy total hero power
enemy_thp_metric = "Enemy_Total_Hero_Power"
enemy_thp_server = 879
enemy_thp_clan_tag_to_filteron = None
enemy_thp_daterange_start = None
enemy_thp_daterange_end = None
enemy_thp_delta_daterange_start = None
enemy_thp_delta_daterange_end = None

####################################################
############## Kills ###############################
####################################################
kills_flag = False # If TRUE, run analysis on kills
kills_metric = "Kills"
kills_server = 870
kills_clan_tag_to_filteron = "GODS"
kills_daterange_start = None
kills_daterange_end = None
kills_delta_daterange_start = "2025-09-27"
kills_delta_daterange_end = "2025-10-07"

####################################################
############## Donations ###########################
####################################################
donations_flag = False # If TRUE, run analysis on donations
donations_metric = "Donations"
donations_server = 870
donations_clan_tag_to_filteron = "GODS"
donations_daterange_start = None
donations_daterange_end = None

####################################################
############## VS ##################################
####################################################
vs_flag = False # If TRUE, run analysis on VS
vs_metric = "VS"
vs_server = 870
vs_clan_tag_to_filteron = "GODS"
vs_daterange_start = None
vs_daterange_end = None

####################################################
############## SERVER WARS: ACTUAL WAR DATA ########
####################################################
serverwars_flag = False # If TRUE, run analysis on serverwars
serverwars_metric = "Server_Wars"
serverwars_server = 870
serverwars_enemy_server = 895
serverwars_clan_tag_to_filteron = "GODS"
serverwars_daterange_start = None
serverwars_daterange_end = None

####################################################
############## SERVER WARS: THP COMPARISON #########
####################################################
serverwars_thp_comparison_flag = False # If TRUE, run analysis on serverwars
serverwars_thp_comparison_metric = "Server_Wars_THP_Comparison"
serverwars_thp_comparison_server = 870
serverwars_thp_comparison_enemy_server = 879
serverwars_thp_comparison_clan_tag_to_filteron = "GODS"
serverwars_thp_comparison_daterange_start = "2025-10-20"
serverwars_thp_comparison_daterange_end = "2025-10-20"



class Globals:

    DEFAULTS = {
        "min_score_filter": None,
        "metric_name": None,
        "server": None,
        "enemy_server": None,
        "clan_tag_to_filteron": None,
        "daterange_start": None,
        "daterange_end": None,
        }
    
    # current state (can be modified)
    variables_state = DEFAULTS.copy()
    
    @classmethod
    def reset(cls):
        cls.variables_state = cls.DEFAULTS.copy()
    
    # Directory of the script
    try:
        SCRIPT_DIR = Path(__file__).resolve().parent
    except NameError:
        # __file__ is not defined (e.g., Jupyter)
        SCRIPT_DIR = Path.cwd()
    os.chdir(SCRIPT_DIR)  # optional, only if you want the working dir changed
    
    
    # === CONFIG ===
    REPORT_TYPES = ["vs", "thp", "donations", "kills", "enemy_thp", "server_wars"]
    TABLE_NAME = "reports"
    UNIQUE_KEYS = ["date", "time", "time_zone", "server", "enemy_server", "overall_rank", "clan_tag", "name"]

    
    
    # Build file paths relative to script
    BASE_DIR = SCRIPT_DIR / "Processed_Data/Verified_Data"  # root of all verified report folders
    DB_FILE_NAME = "server_reports.db"
    DB_PATH = BASE_DIR / DB_FILE_NAME
    PROCESSED_LOG_FILE_NAME = "processed_files.txt"
    PROCESSED_LOG = BASE_DIR / PROCESSED_LOG_FILE_NAME
    
    ALLIANCE_REFERENCE_LIST_NAME = "alliance_reference_list.csv"
    ALLIANCE_REFERENCE_LIST_PATH = SCRIPT_DIR / ALLIANCE_REFERENCE_LIST_NAME
    USERNAME_REFERENCE_LIST_NAME = "username_reference_list.csv"
    USERNAME_REFERENCE_LIST_PATH = SCRIPT_DIR / USERNAME_REFERENCE_LIST_NAME
    
    OUTPUT_DIRECTORY = "Processed_Data"
    OUTPUT_DIRECTORY_PATH = SCRIPT_DIR / OUTPUT_DIRECTORY
    SUMMARY_REPORT_FILES_OUTPUT_DIRECTORY = "Summary_Reports"
    SUMMARY_REPORT_FILES_OUTPUT_DIRECTORY_PATH = OUTPUT_DIRECTORY_PATH / SUMMARY_REPORT_FILES_OUTPUT_DIRECTORY
    PROCESSED_IMG_OUTPUT_DIRECTORY = "Processed_Images"
    PROCESSED_IMG_OUTPUT_DIRECTORY_PATH = SCRIPT_DIR / PROCESSED_IMG_OUTPUT_DIRECTORY
    
    NUMBER_PATTERN_USERNAME_WITH_CLANTAG = re.compile(
    r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)'
    )
    
    NUMBER_PATTERN_USERNAME_ONLY = re.compile(
    r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\s*$'
    )
    
    NUMBERS_FILTERED_1 = r"(\d{1,3}(?:,\d{3})+)"
    
    NUMBERS_FILTERED_2 = r"(\d{1,3}(?:,\d{3})+|\d+)"

    OPTIMIZATION_THRESHOLD = 95
    MAX_OPTI_STEPS_WITHOUT_IMPROVEMENT = 20
    INTERACTIVE_OPTIMIZATION_THRESHOLD = 90  # Adjust as needed
    
    MIN_SCORE_THP = 10_000_000
    MIN_SCORE_KILLS = 50_000
    MIN_SCORE_DONATIONS = 1_000
    MIN_SCORE_VS = 50_000
    MIN_SCORE_SERVERWARS = 100_000

def rebuild_database_from_scratch():
    conn = sqlite3.connect(Globals.DB_PATH)
    conn.execute("DROP TABLE IF EXISTS reports;")  # delete the table completely
    conn.commit()
    conn.close()
    
    # Clear the processed files log so all CSVs are treated as new
    with open(Globals.PROCESSED_LOG, "w") as f:
        f.write("")
    
    print("✅ Database table and processed files log cleared.")



def close_database():
    # Try to clean up connections
    try:
        conn = sqlite3.connect(Globals.DB_PATH)
        conn.close()
    except:
        pass
    
    gc.collect()  # force garbage collection to clear hidden references
    print("✅ All SQLite connections closed (hopefully).")


# === SETUP HELPERS ===
def ensure_directories():
    """Create BASE_DIR and expected subfolders if they don't exist."""
    Globals.BASE_DIR.mkdir(parents=True, exist_ok=True)
    for folder in Globals.REPORT_TYPES:
        (Globals.BASE_DIR / folder).mkdir(exist_ok=True)
    print(f"📁 Verified directory structure under: {Globals.BASE_DIR}")


def load_processed_files():
    """Return a set of files already processed."""
    if not Globals.PROCESSED_LOG.exists():
        return set()
    with open(Globals.PROCESSED_LOG, "r") as f:
        return set(line.strip() for line in f)


def save_processed_files(files):
    """Append processed file names to the log."""
    with open(Globals.PROCESSED_LOG, "a") as f:
        for file in files:
            f.write(f"{file}\n")


def ensure_table_exists(conn):
    """Create the database table if it doesn't exist."""
    conn.execute(f"""
    CREATE TABLE IF NOT EXISTS {Globals.TABLE_NAME} (
        date TEXT,
        time TEXT,
        time_zone TEXT,
        server INTEGER,
        enemy_server INTEGER,
        overall_rank INTEGER,
        clan_tag TEXT,
        extracted_name TEXT,
        matched_name TEXT,
        name TEXT,
        match_score FLOAT,
        score REAL,
        report_type TEXT
    )
    """)


def find_report_files():
    """Find all CSV files in subdirectories (vs, thp, donations, etc.)"""
    return list(Globals.BASE_DIR.rglob("*.csv"))


def get_report_type(file_path):
    """Infer report type from its parent folder name."""
    return file_path.parent.name.lower()


def update_database():
    """
    Load new CSVs into the SQLite database, avoiding duplicates.
    Fully robust: handles path normalization, empty CSVs, and debug prints.
    """

    ensure_directories()

    # --- Load list of already processed files ---
    processed = set(Path(f).resolve() for f in load_processed_files())

    # --- Find all CSV files recursively ---
    all_files = [f.resolve() for f in find_report_files()]
    new_files = [f for f in all_files if f not in processed]

    print(f"📁 Total CSV files found: {len(all_files)}")
    print(f"🆕 New files to process: {[f.name for f in new_files]}")

    # --- Connect to DB and ensure table exists ---
    conn = sqlite3.connect(Globals.DB_PATH)
    ensure_table_exists(conn)

    if not new_files:
        print("✅ No new reports found.")
        conn.close()
        return

    added_files = []

    for file in new_files:
        try:
            df = pd.read_csv(file)
            if df.empty:
                print(f"⚠️ Skipping empty file: {file.name}")
                continue

            # Normalize column names
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

            # Add report_type based on parent folder
            df["report_type"] = get_report_type(file)

            # Convert date column to string
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce").astype(str)

            # --- Ensure expected columns and consistent dtypes before merge ---
            expected_columns = [
                "date", "time", "time_zone", "server", "enemy_server", "overall_rank",
                "clan_tag", "extracted_name", "matched_name", "name",
                "match_score", "score", "report_type"
            ]
            for col in expected_columns:
                if col not in df.columns:
                    df[col] = None
            
            # Coerce numeric columns to proper dtype
            for col in ["server", "enemy_server", "overall_rank"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")  # nullable integer
            
            # Ensure text columns are strings (avoids mixed dtypes)
            for col in ["clan_tag", "extracted_name", "matched_name", "name", "time_zone"]:
                df[col] = df[col].astype("string")


            # --- Duplicate detection based on UNIQUE_KEYS ---
            if Path(Globals.DB_PATH).exists():
                existing = pd.read_sql_query(
                    f"SELECT {', '.join(Globals.UNIQUE_KEYS)} FROM {Globals.TABLE_NAME}", conn
                )
                
                # 🔧 Match dtypes with df to avoid merge errors
                for col in ["server", "enemy_server", "overall_rank"]:
                    if col in existing.columns:
                        existing[col] = pd.to_numeric(existing[col], errors="coerce").astype("Int64")
                for col in ["clan_tag", "extracted_name", "matched_name", "name", "time_zone"]:
                    if col in existing.columns:
                        existing[col] = existing[col].astype("string")

                if not existing.empty:
                    df = df.merge(existing, on=Globals.UNIQUE_KEYS, how="left", indicator=True)
                    df = df[df["_merge"] == "left_only"].drop(columns="_merge")

            if df.empty:
                print(f"⚠️ All rows in {file.name} already exist — skipped.")
                continue

            # --- Ensure expected columns and clean types ---
            expected_columns = [
                "date", "time", "time_zone", "server", "enemy_server", "overall_rank",
                "clan_tag", "extracted_name", "matched_name", "name",
                "match_score", "score", "report_type"
            ]
            for col in expected_columns:
                if col not in df.columns:
                    df[col] = None
            
            # Force enemy_server to numeric with nulls instead of blanks
            df["enemy_server"] = pd.to_numeric(df.get("enemy_server", pd.Series()), errors="coerce")
            
            # --- Insert into database ---
            df.to_sql(Globals.TABLE_NAME, conn, if_exists="append", index=False)

            print(f"📥 Added {len(df)} rows from {file.name} ({df['report_type'].iloc[0]})")
            added_files.append(str(file.resolve()))

        except Exception as e:
            print(f"❌ Error processing {file.name}: {e}")

    conn.close()

    # --- Save processed files to log ---
    if added_files:
        save_processed_files(added_files)
        print(f"✅ Added {len(added_files)} new file(s) to database.")



def generate_summary():
    """Generate quick statistics and plots."""
    if not Globals.DB_PATH.exists():
        print("⚠️ Database not found. Run update_database() first.")
        return

    conn = sqlite3.connect(Globals.DB_PATH)
    df = pd.read_sql_query(f"SELECT * FROM {Globals.TABLE_NAME}", conn)
    conn.close()

    print(f"\n📊 Loaded {len(df):,} total rows from database.")
    print(f"🧾 Report types found: {sorted(df['report_type'].unique())}")

    # Example summary stats
    print("\nAverage score by report type:")
    print(df.groupby("report_type")["score"].mean().round(2).sort_values())

    print("\nTop 5 users by average score:")
    print(df.groupby("extracted_name")["score"].mean().sort_values(ascending=False).head(5))

    # Example visualization
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x="report_type", y="score")
    plt.title("Score Distribution by Report Type")
    plt.tight_layout()
    plt.savefig("score_distribution_by_type.png")
    plt.close()
    print("📈 Saved plot: score_distribution_by_type.png")
 
    
def generate_thp_comparison_server_wars(normalize=False):
    """
    Compare THP vs Enemy THP distributions for server and enemy_server
    over a specified date range.
    """
    server = Globals.variables_state["server"]
    enemy_server = Globals.variables_state["enemy_server"]
    date_start = Globals.variables_state["daterange_start"]
    date_end = Globals.variables_state["daterange_end"]

    conn = sqlite3.connect(Globals.DB_PATH)

    result = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (Globals.TABLE_NAME,)
    ).fetchone()
    if not result:
        print(f"⚠️ Table '{Globals.TABLE_NAME}' not found in {Globals.DB_PATH}. Run update_database() first.")
        conn.close()
        return

    print(f"📅 Using date range: {date_start} → {date_end}")

    # --- Query THP data for both servers within range ---
    query_server = f"""
    SELECT *
    FROM {Globals.TABLE_NAME}
    WHERE report_type = 'thp'
      AND server = {server}
      AND date = (
          SELECT MAX(date)
          FROM {Globals.TABLE_NAME}
          WHERE report_type = 'thp'
            AND server = {server}
            AND date BETWEEN ? AND ?
      )
    """
    
    query_enemy_server = f"""
    SELECT *
    FROM {Globals.TABLE_NAME}
    WHERE report_type = 'enemy_thp'
      AND enemy_server = {enemy_server}
      AND date = (
          SELECT MAX(date)
          FROM {Globals.TABLE_NAME}
          WHERE report_type = 'enemy_thp'
            AND enemy_server = {enemy_server}
            AND date BETWEEN ? AND ?
      )
    """


    df_server = pd.read_sql_query(query_server, conn, params=[date_start, date_end])
    df_enemy_server = pd.read_sql_query(query_enemy_server, conn, params=[date_start, date_end])
    conn.close()
    
    # Get latest available dates for both sides
    latest_server_date = df_server["date"].max() if not df_server.empty else "N/A"
    latest_enemy_date = df_enemy_server["date"].max() if not df_enemy_server.empty else "N/A"


    if df_server.empty and df_enemy_server.empty:
        print("⚠️ No THP data found in specified date range.")
        return

    # --- Combine scores ---
    scores_server = df_server["score"].dropna()
    scores_enemy_server = df_enemy_server["score"].dropna()
    
    max_score = max(scores_server.max() if not scores_server.empty else 0,
                scores_enemy_server.max() if not scores_enemy_server.empty else 0)
    



    # --- Define bins ---
    bin_width = 5_000_000
    try:
        min_bin = (np.floor(min(scores_server.min(), scores_enemy_server.min()) / bin_width) * bin_width).astype(int)
    except:
        min_bin = 90_000_000  # or scores_870.min() if you want fully dynamic

    max_bin = (np.ceil(max_score / bin_width) * bin_width).astype(int)
    
    bins = np.arange(min_bin, max_bin + bin_width, bin_width)


    # --- Optional normalization (percent of total) ---
    if normalize:
        weights_server = np.ones_like(scores_server) / len(scores_server) if len(scores_server) > 0 else None
        weights_enemy_server = np.ones_like(scores_enemy_server) / len(scores_enemy_server) if len(scores_enemy_server) > 0 else None
        counts_server, edges = np.histogram(scores_server, bins=bins, weights=weights_server)
        counts_enemy_server, _ = np.histogram(scores_enemy_server, bins=bins, weights=weights_enemy_server)
        ylabel = "Percentage of Players"
    else:
        counts_server, edges = np.histogram(scores_server, bins=bins)
        counts_enemy_server, _ = np.histogram(scores_enemy_server, bins=bins)
        ylabel = "Number of Players"

    # --- Align bins ---
    min_len = min(len(counts_server), len(counts_enemy_server))
    counts_server = counts_server[:min_len]
    counts_enemy_server = counts_enemy_server[:min_len]
    bin_centers = edges[:-1][:min_len] + 2_500_000
    bar_width = 2_000_000

    # --- Plot ---
    plt.figure(figsize=(12, 6))
    plt.bar(bin_centers - bar_width/2, counts_server, width=bar_width,
            color="blue", alpha=0.7, label=f"Server {server}")
    plt.bar(bin_centers + bar_width/2, counts_enemy_server, width=bar_width,
            color="orange", alpha=0.7, label=f"Server {enemy_server}")

    plt.title(
        f"THP Comparison — Server {server} ({latest_server_date[:10]}) vs "
        f"Server {enemy_server} ({latest_enemy_date[:10]})",
        fontsize=14
    )
    
    # Add vertical mean line
    mean_score_server = df_server["score"].mean()
    plt.axvline(mean_score_server, color="blue", linestyle="--", linewidth=1.5, label=f"Mean: {mean_score_server:,.0f}")
    mean_score_enemy_server = df_enemy_server["score"].mean()
    plt.axvline(mean_score_enemy_server, color="orange", linestyle="--", linewidth=1.5, label=f"Mean: {mean_score_enemy_server:,.0f}")
    

    plt.xlabel("THP Score")
    plt.ylabel(ylabel)
    plt.legend()
    plt.xlim(min_bin, max_bin)
    plt.minorticks_on()
    plt.tick_params(axis='x', which='minor', length=0)  # removes tick marks
    plt.grid(which='major', axis='y', linestyle='-', alpha=0.6)  # major grid
    plt.grid(which='minor', axis='y', linestyle=':', alpha=0.3)   # minor grid
    #plt.grid(which='minor', axis='x', linestyle=':', alpha=0.0)   # minor grid



    # --- Bin range labels ---
    labels = [f"{int(b/1_000_000)}M–{int((b+5_000_000)/1_000_000)}M" for b in edges[:-1]]
    plt.xticks(bin_centers[::1], labels[::1], rotation=90, ha='center', fontsize=9)

    plt.tight_layout()
    out_path = Globals.SUMMARY_REPORT_FILES_OUTPUT_DIRECTORY_PATH / f"thp_distribution_{server}_vs_{enemy_server}_{date_start}_to_{date_end}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()


def generate_thp_comparison_top_alliances(metric_type, server, clan_tag, time_stamp, date_start=None, date_end=None):
    """
    Compare THP in top alliances for server
    over a specified date range.
    """
    # server = Globals.variables_state["server"]
    # enemy_server = Globals.variables_state["enemy_server"]
    # date_start = Globals.variables_state["daterange_start"]
    # date_end = Globals.variables_state["daterange_end"]

    conn = sqlite3.connect(Globals.DB_PATH)

    result = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (Globals.TABLE_NAME,)
    ).fetchone()
    if not result:
        print(f"⚠️ Table '{Globals.TABLE_NAME}' not found in {Globals.DB_PATH}. Run update_database() first.")
        conn.close()
        return

    print(f"📅 Using date range: {date_start} → {date_end}")
    
    # --- Handle None date range by defaulting to latest available date ---
    if date_start is None or date_end is None:
        latest_date_row = pd.read_sql_query(f"""
            SELECT MAX(date) AS max_date
            FROM {Globals.TABLE_NAME}
            WHERE report_type = ?
              AND ({'enemy_server' if metric_type == 'enemy_thp' else 'server'}) = ?;
        """, conn, params=[metric_type, server])
    
        latest_date = latest_date_row['max_date'].iloc[0]
    
        if latest_date is None:
            print("⚠️ No data found for this metric/server in database.")
            conn.close()
            return
    
        date_start = latest_date
        date_end = latest_date
    
    print(f"📅 Adjusted date range: {date_start} → {date_end}")

    
    if metric_type == "thp":
        
        # --- Query THP data for both servers within range ---
        query_server = f"""
        SELECT *
        FROM {Globals.TABLE_NAME}
        WHERE report_type = 'thp'
          AND server = {server}
          AND date = (
              SELECT MAX(date)
              FROM {Globals.TABLE_NAME}
              WHERE report_type = 'thp'
                AND server = {server}
                AND date BETWEEN ? AND ?
          )
        """
        
    elif metric_type == "enemy_thp":      
        
        # --- Query enemy THP data for both servers within range ---
        query_server = f"""
        SELECT *
        FROM {Globals.TABLE_NAME}
        WHERE report_type = 'enemy_thp'
          AND enemy_server = {server}
          AND date = (
              SELECT MAX(date)
              FROM {Globals.TABLE_NAME}
              WHERE report_type = 'enemy_thp'
                AND enemy_server = {server}
                AND date BETWEEN ? AND ?
          )
        """


    df_server = pd.read_sql_query(query_server, conn, params=[date_start, date_end])
    conn.close()
    
    # Get latest available dates for both sides
    latest_server_date = df_server["date"].max() if not df_server.empty else "N/A"

    if df_server.empty:
        print("⚠️ No THP data found in specified date range.")
        return

    # --- Get top 4 alliances with most number of players in top 200 THP ---
    top_four_alliances = df_server['clan_tag'].value_counts().head(4)
    top_four_alliances = top_four_alliances.reset_index()

    # --- Combine scores ---
    scores_server = df_server["score"].dropna()
    scores_server_total = np.sum(scores_server)
    
    df_alliance1 = df_server.loc[df_server["clan_tag"] == top_four_alliances.loc[0,"clan_tag"]]
    scores_server_alliance1 = df_alliance1["score"].dropna()
    scores_server_alliance1_total = np.sum(scores_server_alliance1)
    
    df_alliance2 = df_server.loc[df_server["clan_tag"] == top_four_alliances.loc[1,"clan_tag"]]
    scores_server_alliance2 = df_alliance2["score"].dropna()
    scores_server_alliance2_total = np.sum(scores_server_alliance2)
    
    df_alliance3 = df_server.loc[df_server["clan_tag"] == top_four_alliances.loc[2,"clan_tag"]]
    scores_server_alliance3 = df_alliance3["score"].dropna()
    scores_server_alliance3_total = np.sum(scores_server_alliance3)
    
    df_alliance4 = df_server.loc[df_server["clan_tag"] == top_four_alliances.loc[3,"clan_tag"]]
    scores_server_alliance4 = df_alliance4["score"].dropna()
    scores_server_alliance4_total = np.sum(scores_server_alliance4)
    
    df_solo_players = df_server.loc[df_server["clan_tag"] == "No_Clan"]
    scores_solo_players = df_solo_players["score"].dropna()
    scores_solo_players_total = np.sum(scores_solo_players)
    
    num_players_top_4_alliances = len(scores_server_alliance1) + len(scores_server_alliance2) + len(scores_server_alliance3) + len(scores_server_alliance4)
    
    num_players_restofserver = len(df_server) - num_players_top_4_alliances - len(df_solo_players)
    scores_players_restofserver_total = (scores_server_total
                                         - scores_solo_players_total
                                         - scores_server_alliance1_total
                                         - scores_server_alliance2_total
                                         - scores_server_alliance3_total
                                         - scores_server_alliance4_total
                                         )
    
    max_score = scores_server.max()

    # --- Define bins ---
    bin_width = 5_000_000
    try:
        min_bin = (np.floor(scores_server.min() / bin_width) * bin_width).astype(int)
    except:
        min_bin = 90_000_000  # or scores_870.min() if you want fully dynamic

    max_bin = (np.ceil(max_score / bin_width) * bin_width).astype(int)
    
    bins = np.arange(min_bin, max_bin + bin_width, bin_width)

    counts_server_alliance1, edges = np.histogram(scores_server_alliance1, bins=bins)
    counts_server_alliance2, _ = np.histogram(scores_server_alliance2, bins=bins)
    counts_server_alliance3, _ = np.histogram(scores_server_alliance3, bins=bins)
    counts_server_alliance4, _ = np.histogram(scores_server_alliance4, bins=bins)
    
    #counts_enemy_server, _ = np.histogram(scores_enemy_server, bins=bins)
    ylabel = "Number of Players"

    # --- Align bins ---
    min_len = min(len(counts_server_alliance1), len(counts_server_alliance2),len(counts_server_alliance3), len(counts_server_alliance4))
    counts_server_alliance1 = counts_server_alliance1[:min_len]
    counts_server_alliance2 = counts_server_alliance2[:min_len]
    counts_server_alliance3 = counts_server_alliance3[:min_len]
    counts_server_alliance4 = counts_server_alliance4[:min_len]

    bin_centers = edges[:-1][:min_len] + bin_width * (1/2)
    bar_width = (bin_width / 4) * 0.9

    # --- Plot ---
    plt.figure(figsize=(24, 6))
    plt.bar(bin_centers - bar_width*(3/2), counts_server_alliance1, width=bar_width,
            color="royalblue", alpha=0.7, label=f"{top_four_alliances.loc[0,"clan_tag"]} ({len(scores_server_alliance1)} players, {int(np.round(scores_server_alliance1_total / 1_000_000, 0))}M THP)")
    plt.bar(bin_centers - bar_width*(1/2), counts_server_alliance2, width=bar_width,
            color="orange", alpha=0.7, label=f"{top_four_alliances.loc[1,"clan_tag"]} ({len(scores_server_alliance2)} players, {int(np.round(scores_server_alliance2_total / 1_000_000, 0))}M THP)")
    plt.bar(bin_centers + bar_width*(1/2), counts_server_alliance3, width=bar_width,
            color="red", alpha=0.7, label=f"{top_four_alliances.loc[2,"clan_tag"]} ({len(scores_server_alliance3)} players, {int(np.round(scores_server_alliance3_total / 1_000_000, 0))}M THP)")
    plt.bar(bin_centers + bar_width*(3/2), counts_server_alliance4, width=bar_width,
            color="black", alpha=0.7, label=f"{top_four_alliances.loc[3,"clan_tag"]} ({len(scores_server_alliance4)} players, {int(np.round(scores_server_alliance4_total / 1_000_000, 0))}M THP)")


    plt.title(
        f"THP Comparison — Server {server} - Top 200 Players - Alliance Distributions - ({latest_server_date[:10]}))",
        fontsize=14
    )


    plt.xlabel("THP Score")
    plt.ylabel(ylabel)
    plt.legend()
    ax = plt.gca()
    # Create a proxy artist for the additional text
    proxy_artist = mpatches.Rectangle((0, 0), 1, 1, fc="none", ec="none", lw=0)
    
    # Get existing handles and labels
    handles, labels_lengend = ax.get_legend_handles_labels()
    
    # Add the proxy artist and its label to the lists
    handles.append(proxy_artist)
    
    
    #{int(np.round(scores_solo_players_total / 1_000_000, 0))}M THP)"
    labels_lengend.append(f"{len(df_solo_players)} No Clan Players ({int(np.round(scores_solo_players_total / 1_000_000, 0))}M THP)\n{num_players_restofserver} other {server} Players ({int(np.round(scores_players_restofserver_total / 1_000_000, 0))}M THP)")

    # Create the legend with the combined handles and labels
    plt.legend(handles, labels_lengend)
    
    plt.xlim(min_bin, max_bin)
    plt.minorticks_on()
    plt.tick_params(axis='x', which='minor', length=0)  # removes tick marks
    plt.grid(which='major', axis='y', linestyle='-', alpha=0.6)  # major grid
    plt.grid(which='minor', axis='y', linestyle=':', alpha=0.3)   # minor grid
    #plt.grid(which='minor', axis='x', linestyle=':', alpha=0.0)   # minor grid
    ax = plt.gca()
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_ticks_position("both") 
    ax.tick_params(axis='y', which='both', labelleft=True, labelright=True, left=True, right=True)



    # --- Bin range labels ---
    labels = [f"{int(b/1_000_000)}M–{int((b+5_000_000)/1_000_000)}M" for b in edges[:-1]]
    plt.xticks(bin_centers[::1], labels[::1], rotation=90, ha='center', fontsize=9)

    #plt.tight_layout()
    out_path = Globals.SUMMARY_REPORT_FILES_OUTPUT_DIRECTORY_PATH / f"thp_distribution_{server}_Top_4_Alliances_{date_start}_to_{date_end}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

def generate_metric_summary(metric_type, server, clan_tag, time_stamp, date_start=None, date_end=None):
    """
    Generate a leaderboard + histogram side by side for a given metric type (e.g. 'thp', 'kills', etc.)
    and save as a single-page summary image.
    """
    conn = sqlite3.connect(Globals.DB_PATH)

    # === Handle date range logic ===
    if date_start and date_end:
        # Use explicit range
        date_filter = "AND date BETWEEN ? AND ?"
        params = [metric_type, server, clan_tag, date_start, date_end]
        query = f"""
            SELECT *
            FROM {Globals.TABLE_NAME}
            WHERE report_type = ?
              AND server = ?
              AND clan_tag = ?
              {date_filter}
        """
    else:
        # Default: only use the most recent available date
        query = f"""
            SELECT *
            FROM {Globals.TABLE_NAME}
            WHERE report_type = ?
              AND server = ?
              AND clan_tag = ?
              AND date = (
                  SELECT MAX(date)
                  FROM {Globals.TABLE_NAME}
                  WHERE report_type = ?
                    AND server = ?
                    AND clan_tag = ?
              )
        """
        params = [metric_type, server, clan_tag, metric_type, server, clan_tag]

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()


    if df.empty:
        print(f"⚠️ No data found for {metric_type.upper()} ({server}, {clan_tag}).")
        return

    latest_date = df["date"].max()

    # === Compute stats ===
    total_players = len(df)
    avg_score = df["score"].mean()
    med_score = df["score"].median()
    min_score = df["score"].min()
    max_score = df["score"].max()

    # --- All-time stats for this metric & server ---
    conn = sqlite3.connect(Globals.DB_PATH)
    all_time_query = f"""
        SELECT name, score, date
        FROM {Globals.TABLE_NAME}
        WHERE report_type = ?
          AND server = ?
          AND clan_tag = ?
    """
    all_time_df = pd.read_sql_query(all_time_query, conn, params=[metric_type, server, clan_tag])
    conn.close()
    
    if not all_time_df.empty:
        all_time_avg = all_time_df["score"].mean()
        all_time_max_row = all_time_df.loc[all_time_df["score"].idxmax()]
        all_time_max = all_time_max_row["score"]
        all_time_max_name = all_time_max_row["name"]
        all_time_max_date = all_time_max_row["date"]  # <-- add this
    else:
        all_time_avg = all_time_max = 0
        all_time_max_name = ""
        all_time_max_date = ""



    top_players = df.nlargest(10, "score")[["name", "score"]].copy()
    top_players["score"] = top_players["score"].map(lambda x: f"{x:,.0f}")

    print(f"\n📊 Summary for {metric_type.upper()} — Server {server} ({clan_tag}) on {latest_date}:")
    print(top_players.to_string(index=False))

    # === Build Plot Layout ===
    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 8.5), gridspec_kw={'width_ratios': [2, 1]})
    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 9), gridspec_kw={'width_ratios': [2, 1]})


    # After plotting everything
    # gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.1)  # histogram twice as wide
    
    # ax1 = fig.add_subplot(gs[0, 0])
    # ax2 = fig.add_subplot(gs[0, 1])
    
    fig.suptitle(f"{metric_type.upper()} Summary — Server {server} ({clan_tag}) [{latest_date[:10]}]", fontsize=14)

    metric_bin_widths = {
        "thp": 2_000_000,
        "enemy_thp": 5_000_000,
        "kills": 1_000_000,
        "donations": 5_000,
        "vs": 10_000_000,
        "server_wars": 500_000,
    }



    # --- Histogram (Left) with Mean Line ---
    scores = df["score"].dropna()
    bin_width = metric_bin_widths.get(metric_type, 10_000)  # default if missing
    min_bin = int(np.floor(scores.min() / bin_width) * bin_width)
    max_bin = int(np.ceil(scores.max() / bin_width) * bin_width)
    bins = np.arange(min_bin, max_bin + bin_width, bin_width)
    
    counts, edges = np.histogram(scores, bins=bins)
    
    bin_centers = edges[:-1] + bin_width / 2
    
    ax1.bar(bin_centers, counts, width=bin_width*0.9, color="royalblue", alpha=0.7)
    ax1.set_xlabel(metric_type.upper())
    ax1.set_ylabel("Number of Players")
    ax1.set_title("Score Distribution")
    ax1.grid(axis="y", linestyle="--", alpha=0.6)
    # Determine label scaling and unit dynamically
    if max_bin >= 1_000_000:
        scale = 1_000_000
        unit = "M"
    elif max_bin >= 1_000:
        scale = 1_000
        unit = "K"
    else:
        scale = 1
        unit = ""
    
    # Create human-friendly bin range labels
    labels = [f"{b/scale:.1f}{unit}–{(b+bin_width)/scale:.1f}{unit}" for b in edges[:-1]]
    ax1.set_xticks(bin_centers)
    ax1.set_xticklabels(labels, rotation=90, ha="center", fontsize=9)

    #labels = [f"{int(b/1_000_000)}M–{int((b+bin_width)/1_000_000)}M" for b in edges[:-1]]  # for THP
    ax1.set_xticks(bin_centers)
    ax1.set_xticklabels(labels, rotation=90, ha="center", fontsize=9)
    ax1.grid(which='major', axis='y', linestyle='-', alpha=0.6)
    ax1.grid(which='minor', axis='y', linestyle=':', alpha=0.3)
    ax1.minorticks_on()
    ax1.tick_params(axis='x', which='minor', length=0)  # remove minor x ticks


    
    # Add vertical mean line
    mean_score = df["score"].mean()
    ax1.axvline(mean_score, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_score:,.0f}")
    
    # Add legend for mean line
    ax1.legend()

    
    # --- Leaderboard (Right) ---
    ax2.axis("off")
    
    # Sort by score descending to get full ranking
    df_sorted = df.sort_values("score", ascending=False).reset_index(drop=True)
    
    # Top 10
    if len(df_sorted) < 10:
        num_players_head_tail = len(df_sorted)
    else:
        num_players_head_tail = 10
    top_players = df_sorted.head(num_players_head_tail).copy()
    top_players["score"] = top_players["score"].map(lambda x: f"{x:,.0f}")
    
    # Bottom 10
    bottom_players = df_sorted.tail(num_players_head_tail).copy()
    bottom_players["score"] = bottom_players["score"].map(lambda x: f"{x:,.0f}")
    
    # # Build table: show top 10 first, blank row, then bottom 10
    # table_data = [["Rank", "Name", "Score"]]
    
    if metric_type == "thp" or metric_type == "server_wars":
        server_rank_flag = True
        # Build table: show top 10 first, blank row, then bottom 10
        table_data = [["Server_Rank", "Alliance_Rank", "Name", "Score"]]
    else:
        server_rank_flag = False
        # Build table: show top 10 first, blank row, then bottom 10
        table_data = [["Alliance_Rank", "Name", "Score"]]
    
    # Top 10
    for i, row in enumerate(top_players.itertuples(index=False), start=1):
        if server_rank_flag == True:
            server_rank = row.overall_rank
            alliance_rank = i
            if not row.overall_rank or np.isnan(row.overall_rank):
                table_data.append(["U.R.", "U.R.", row.name, row.score])
            else: 
                table_data.append([int(server_rank), alliance_rank, row.name, row.score])
        if server_rank_flag == False:
            alliance_rank = i
            if not row.overall_rank or np.isnan(row.overall_rank):
                table_data.append(["U.R.", row.name, row.score])
            else: 
                table_data.append([alliance_rank, row.name, row.score])

    # Blank row
    if server_rank_flag == True:
        table_data.append(["", "", "", ""])
    else:
        table_data.append(["", "", ""])
    
    # Bottom 10 (use true rank)
    total_players = len(df_sorted)
    for i, row in enumerate(bottom_players.itertuples(index=False), start=total_players - num_players_head_tail + 1):
        if server_rank_flag == True:
            server_rank = row.overall_rank
            alliance_rank = i
            if not row.overall_rank or np.isnan(row.overall_rank):
                table_data.append(["U.R.", "U.R.", row.name, row.score])
            else: 
                table_data.append([int(server_rank), alliance_rank, row.name, row.score])
        if server_rank_flag == False:
            alliance_rank = i
            if not row.overall_rank or np.isnan(row.overall_rank):
                table_data.append(["U.R.", row.name, row.score])
            else: 
                table_data.append([alliance_rank, row.name, row.score])
    
    # Create table
    if server_rank_flag == True:
        table = ax2.table(cellText=table_data, loc="center", cellLoc="center", colWidths=[0.25, 0.25, 0.4, 0.35])
    else:
        table = ax2.table(cellText=table_data, loc="center", cellLoc="center", colWidths=[0.25, 0.4, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.5, 1.5)

    


    # --- Bottom footer: current summary stats ---
    fig.text(
        0.5, 0.04,  # slightly above the very bottom
        f"Total Players: {total_players} | Avg: {avg_score:,.0f} | Median: {med_score:,.0f} | Min: {min_score:,.0f} | Max: {max_score:,.0f}",
        ha="center", fontsize=12, color="black"
    )
    
    # --- Bottom footer line 2: all-time stats ---
    fig.text(
        0.5, 0.01,  # x=0.01, y=0.01 is safe margin from bottom-left
        f"All-Time Avg: {all_time_avg:,.0f} | All-Time Max: {all_time_max:,.0f} "
        f"({all_time_max_name} on {all_time_max_date})",
        ha="center", fontsize=12, color="black"
    )



    
    # gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.1)  # histogram twice as wide
    
    # ax1 = fig.add_subplot(gs[0, 0])
    # ax2 = fig.add_subplot(gs[0, 1])
    # --- Save the combined summary ---
    plt.tight_layout(rect=[0, 0.07, 1, 0.97])
    #plt.show()
    out_path = Globals.SUMMARY_REPORT_FILES_OUTPUT_DIRECTORY_PATH / f"{metric_type}_summary_server_{server}_{latest_date[:10]}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"📈 Saved combined summary image: {out_path}")

def generate_metric_change_summary(metric_type, server, clan_tag, time_stamp, date_start=None, date_end=None, delta_date_start=None, delta_date_end=None):
    """
    Generates a 'change since last report' summary for the given metric.
    Calculates difference in scores per player between the two latest reports,
    includes new/missing players, and shows percentage change.
    """
    conn = sqlite3.connect(Globals.DB_PATH)

    date_filter = ""
    params = []
    if date_start and date_end:
        date_filter = "AND date BETWEEN ? AND ?"
        params = [date_start, date_end]

    query = f"""
        SELECT name, date, score
        FROM {Globals.TABLE_NAME}
        WHERE report_type = ?
          AND server = ?
          AND clan_tag = ?
          {date_filter}
    """

    df = pd.read_sql_query(query, conn, params=[metric_type, server, clan_tag, *params])
    conn.close()

    if df.empty or df["date"].nunique() < 2:
        print(f"⚠️ Not enough data to calculate change for {metric_type.upper()} ({server}, {clan_tag}).")
        return

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["name", "date"])

    # --- Get latest and previous report dates ---
    unique_dates = sorted(df["date"].unique())
    latest_date = unique_dates[-1]
    previous_date = unique_dates[-2]

    df_latest = df[df["date"] == latest_date].copy()
    df_prev = df[df["date"] == previous_date].copy()

    # --- Merge dataframes ---
    merged = pd.merge(df_latest, df_prev, on="name", suffixes=("_current", "_prev"), how="outer")
    # Flatten tuple-like entries (e.g., (123456,) -> 123456)
    for col in ["score_current", "score_prev"]:
        merged[col] = merged[col].apply(
            lambda x: x[0] if isinstance(x, tuple) and len(x) == 1 else x
        )
    
    # Convert to numeric afterward
    merged["score_current"] = pd.to_numeric(merged["score_current"], errors="coerce")
    merged["score_prev"] = pd.to_numeric(merged["score_prev"], errors="coerce")

    # Identify new/missing players
    merged["status"] = "existing"
    merged.loc[merged["score_prev"].isna(), "status"] = "new"
    merged.loc[merged["score_current"].isna(), "status"] = "missing"

    # Compute absolute and percent change
    merged["change"] = merged["score_current"].fillna(0) - merged["score_prev"].fillna(0)
    merged["pct_change"] = np.where(
        merged["score_prev"].fillna(0) > 0,
        merged["change"] / merged["score_prev"] * 100,
        np.nan
    )

    # Drop rows with both NaN scores (edge cases)
    merged = merged.dropna(subset=["score_current", "score_prev"], how="all")

    if merged.empty:
        print(f"⚠️ No overlapping data found between {previous_date.date()} and {latest_date.date()}.")
        return

    # === Stats (existing players only) ===
    valid_changes = merged.loc[merged["status"] == "existing", "change"]
    avg_change = valid_changes.mean() if not valid_changes.empty else 0
    med_change = valid_changes.median() if not valid_changes.empty else 0
    min_change = valid_changes.min() if not valid_changes.empty else 0
    max_change = valid_changes.max() if not valid_changes.empty else 0




    # === Plot ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 8.5), gridspec_kw={'width_ratios': [1, 1]})
    fig.suptitle(
        f"{metric_type.upper()} Change — Server {server} ({clan_tag}) [{previous_date.date()} → {latest_date.date()}]",
        fontsize=14
    )

    # --- Histogram (Left) with Mean Line (Consistent Style) ---
    scores = merged.loc[merged["status"] == "existing","change"].copy()  # or whatever metric column you’re plotting
    
    metric_bin_widths = {
        "thp": 200_000,
        "enemy_thp": 5_000_000,
        "kills": 50_000,
        "donations": 5_000,
        "vs": 10_000_000,
    }
    
    
    
    # --- Histogram (Left) with Mean Line ---
    #scores = df["score"].dropna()
    bin_width = metric_bin_widths.get(metric_type, 10_000)  # default if missing
    min_bin = int(np.floor(scores.min() / bin_width) * bin_width)
    max_bin = int(np.ceil(scores.max() / bin_width) * bin_width)
    bins = np.arange(min_bin, max_bin + bin_width, bin_width)
    
    counts, edges = np.histogram(scores, bins=bins)
    
    bin_centers = edges[:-1] + bin_width / 2
    
    ax1.bar(bin_centers, counts, width=bin_width*0.9, color="royalblue", alpha=0.7)
    ax1.set_xlabel(metric_type.upper())
    ax1.set_ylabel("Number of Players")
    ax1.set_title("Score Distribution")
    ax1.grid(axis="y", linestyle="--", alpha=0.6)
    # Determine label scaling and unit dynamically
    if max_bin >= 5_000_000:
        scale = 1_000_000
        unit = "M"
    elif max_bin >= 1_000_000:
        scale = 1_000
        unit = "K"
    elif max_bin >= 1_000:
        scale = 1_000
        unit = "K"
    else:
        scale = 1
        unit = ""
    
    # Create human-friendly bin range labels
    labels = [f"{b/scale:.1f}{unit}–{(b+bin_width)/scale:.1f}{unit}" for b in edges[:-1]]
    ax1.set_xticks(bin_centers)
    ax1.set_xticklabels(labels, rotation=90, ha="center", fontsize=9)
    
    #labels = [f"{int(b/1_000_000)}M–{int((b+bin_width)/1_000_000)}M" for b in edges[:-1]]  # for THP
    ax1.set_xticks(bin_centers)
    ax1.set_xticklabels(labels, rotation=90, ha="center", fontsize=9)
    ax1.grid(which='major', axis='y', linestyle='-', alpha=0.6)
    ax1.grid(which='minor', axis='y', linestyle=':', alpha=0.3)
    ax1.minorticks_on()
    ax1.tick_params(axis='x', which='minor', length=0)  # remove minor x ticks
    
    
    
    # Add vertical mean line
    #mean_score = df["score"].mean()
    ax1.axvline(avg_change, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {avg_change:,.0f}")
    
    # Add legend for mean line
    ax1.legend()


    # === Leaderboard Table ===
    df_sorted = merged.sort_values("change", ascending=False).reset_index(drop=True)
    top_players = df_sorted.head(10)
    bottom_players = df_sorted.tail(10)

    table_data = [["Rank", "Name", "Δ Change", "Δ%", "Status"]]
    for i, row in enumerate(top_players.itertuples(), start=1):
        delta = f"{'+' if row.change >= 0 else ''}{row.change:,.0f}"
        pct = f"{row.pct_change:,.1f}%" if not pd.isna(row.pct_change) else "-"
        table_data.append([i, row.name, delta, pct, row.status])
    table_data.append(["", "", "", "", ""])
    total = len(df_sorted)
    for i, row in enumerate(bottom_players.itertuples(), start=total - 9):
        delta = f"{'+' if row.change >= 0 else ''}{row.change:,.0f}"
        pct = f"{row.pct_change:,.1f}%" if not pd.isna(row.pct_change) else "-"
        table_data.append([i, row.name, delta, pct, row.status])

    ax2.axis("off")
    table = ax2.table(cellText=table_data, loc="center", cellLoc="center", colWidths=[0.07, 0.35, 0.22, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.25, 1.15)

    # Highlight new/missing players
    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#DDDDDD")
        elif row_idx > 0:
            status_text = table_data[row_idx][4]
            if status_text == "new":
                cell.set_facecolor("#CCFFCC")  # light green
            elif status_text == "missing":
                cell.set_facecolor("#FFCCCC")  # light red


    # === Footer ===
    fig.text(
        0.5, 0.03,
        f"Existing Players: {len(valid_changes)} | Avg Δ: {avg_change:,.0f} | Median Δ: {med_change:,.0f} | "
        f"Min Δ: {min_change:,.0f} | Max Δ: {max_change:,.0f}",
        ha="center", fontsize=11
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    out_path = Globals.SUMMARY_REPORT_FILES_OUTPUT_DIRECTORY_PATH / f"{metric_type}_change_summary_server_{server}_{latest_date.date()}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📈 Saved change summary: {out_path}")




def generate_weekly_summary_pdf(time_stamp):
    """
    Combines all *_summary_server_*.png images into a multi-page PDF.
    Each page = one metric summary image.
    """
    output_filename = Globals.SUMMARY_REPORT_FILES_OUTPUT_DIRECTORY_PATH / f"Weekly_Summary_Report_{time_stamp}.pdf"
    summary_images = list(Path(".").glob("*_summary_server_*.png"))
    if not summary_images:
        print("⚠️ No summary images found. Run generate_metric_summary() for each metric first.")
        return


    c = canvas.Canvas(f"{output_filename}", pagesize=landscape(letter))
    width, height = landscape(letter)


    for img_path in summary_images:
        try:
            # Convert image to RGB (handles PNG transparency)
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                img_io = BytesIO()
                img.save(img_io, format="PNG")
                img_io.seek(0)

                # Scale to fit PDF page
                img_reader = ImageReader(img_io)
                iw, ih = img.size
                aspect = ih / iw
                max_width = width - 50
                max_height = height - 50
                if iw > max_width:
                    iw = max_width
                    ih = iw * aspect
                if ih > max_height:
                    ih = max_height
                    iw = ih / aspect

                x = (width - iw) / 2
                y = (height - ih) / 2

                c.drawImage(img_reader, x, y, iw, ih)
                c.showPage()

        except Exception as e:
            print(f"⚠️ Skipping {img_path}: {e}")

    c.save()
    print(f"📘 Saved combined PDF report: {output_filename}")

def generate_all_metric_summaries():
    """Automatically generate summary images for all enabled metrics."""
    metrics = [
        ("thp", thp_flag, thp_server, thp_clan_tag_to_filteron, thp_daterange_start, thp_daterange_end, thp_delta_daterange_start, thp_delta_daterange_end),
        ("enemy_thp", enemy_thp_flag, enemy_thp_server, enemy_thp_clan_tag_to_filteron, enemy_thp_daterange_start, enemy_thp_daterange_end, None, None),
        ("kills", kills_flag, kills_server, kills_clan_tag_to_filteron, kills_daterange_start, kills_daterange_end, kills_delta_daterange_start, kills_delta_daterange_end),
        ("donations", donations_flag, donations_server, donations_clan_tag_to_filteron, donations_daterange_start, donations_daterange_end, None, None),
        ("vs", vs_flag, vs_server, vs_clan_tag_to_filteron, vs_daterange_start, vs_daterange_end, None, None),
        ("server_wars", serverwars_flag, serverwars_server, serverwars_clan_tag_to_filteron, serverwars_daterange_start, serverwars_daterange_end, None, None),
    ]


    for metric, flag, server, clan, date_start, date_end, delta_date_start, delta_date_end in metrics:
        if flag:
            generate_metric_summary(metric, server, clan, time_stamp, date_start, date_end)
    
    # === CHANGE summaries for THP and KILLS ===
    if thp_flag:
        generate_metric_change_summary("thp", thp_server, thp_clan_tag_to_filteron, time_stamp, thp_delta_daterange_start, thp_delta_daterange_end)
        generate_thp_comparison_top_alliances("thp", thp_server, thp_clan_tag_to_filteron, time_stamp, thp_delta_daterange_start, thp_delta_daterange_end)
    
    if enemy_thp_flag:
        generate_thp_comparison_top_alliances("enemy_thp", enemy_thp_server, enemy_thp_clan_tag_to_filteron, time_stamp, enemy_thp_delta_daterange_start, enemy_thp_delta_daterange_end)
    
    if kills_flag:
        generate_metric_change_summary("kills", kills_server, kills_clan_tag_to_filteron, time_stamp, kills_delta_daterange_start, kills_delta_daterange_end)

if __name__ == "__main__":
    update_database()
    current_datetime = datetime.now()
    time_stamp = current_datetime.strftime("%Y-%m-%d_%I-%M%p")

    # === Generate all metric summaries ===
    generate_all_metric_summaries()

    # === Generate combined weekly report ===
    #generate_weekly_summary_pdf(time_stamp)
    
    # === SERVER WARS COMPARISON ===
    if serverwars_thp_comparison_flag:
        Globals.reset()
        Globals.variables_state.update({
            "metric_name": str(serverwars_thp_comparison_metric),
            "server": serverwars_thp_comparison_server,
            "enemy_server": serverwars_thp_comparison_enemy_server,
            "clan_tag_to_filteron": serverwars_thp_comparison_clan_tag_to_filteron,
            "daterange_start": serverwars_thp_comparison_daterange_start,
            "daterange_end": serverwars_thp_comparison_daterange_end,
        })
        generate_thp_comparison_server_wars()

    close_database()
