# -*- coding: utf-8 -*-
"""
Improved OCR Script for Long Scrolling Screenshots
"""

import cv2
import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS
import re
import pytesseract
from datetime import datetime
from zoneinfo import ZoneInfo  # Python 3.9+
from rapidfuzz import process, fuzz
import random
import time
import tkinter as tk
from PIL import Image, ImageTk
import gc


# Ensure Tesseract is correctly installed
# Modify this path if needed
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


####################################################
############## INPUTS ##############################
####################################################

gc.collect()  # force garbage collection to clear hidden references

####################################################
############## 870 Total Hero Power ################
####################################################
thp_flag = False # If TRUE, run analysis on total hero power
thp_metric = "Total_Hero_Power"
thp_file_name = "870_thp_10-20-25.JPEG"
thp_server = 870
thp_enemy_server = None
thp_num_players = int(200)
thp_clan_tag_location = "next_to" # options: "below", "next_to", None
thp_clan_tag_to_filteron = "GODS"
if thp_clan_tag_location == None:
    thp_default_clan_tag = thp_clan_tag_to_filteron
else:
    thp_default_clan_tag = None
thp_player_order_continuous = True
thp_rank_strip_start = 60
thp_rank_strip_end = 200
thp_pixels_to_remove = 380  #### No. of pixels to remove from left side of img
thp_chunk_height_reduction_factor = 0.0 # amount to reduce total chunk height by, chunk_height_reduction_factor * 0.5 from top and bottom of chunk
thp_export_server_data_flag = True
thp_export_clan_data_flag = True
thp_report_type = "thp"

####################################################
############## Enemy Total Hero Power ##############
####################################################
enemy_thp_flag = True # If TRUE, run analysis on enemy total hero power
enemy_thp_metric = "Enemy_Total_Hero_Power"
enemy_thp_file_name = "914_thp_10-20-25.JPEG"
enemy_thp_server = None
enemy_thp_enemy_server = 914
enemy_thp_num_players = int(200)
enemy_thp_clan_tag_location = "next_to" # options: "below", "next_to", None
enemy_thp_clan_tag_to_filteron = None
if enemy_thp_clan_tag_location == None:
    enemy_thp_default_clan_tag = enemy_thp_clan_tag_to_filteron
else:
    enemy_thp_default_clan_tag = None
enemy_thp_player_order_continuous = True
enemy_thp_rank_strip_start = 60
enemy_thp_rank_strip_end = 215
enemy_thp_pixels_to_remove = 380  #### No. of pixels to remove from left side of img
enemy_thp_chunk_height_reduction_factor = 0.0 # amount to reduce total chunk height by, chunk_height_reduction_factor * 0.5 from top and bottom of chunk
enemy_thp_export_server_data_flag = True
enemy_thp_export_clan_data_flag = False
enemy_thp_report_type = "enemy_thp"

####################################################
############## Kills ###############################
####################################################
kills_flag = False # If TRUE, run analysis on kills
kills_metric = "Kills"
kills_file_name = "kills_10-5-25.JPEG"
kills_server = 870
kills_enemy_server = None
kills_num_players = int(100)
kills_clan_tag_location = None # options: "below", "next_to", None
kills_clan_tag_to_filteron = "GODS"
if kills_clan_tag_location == None:
    kills_default_clan_tag = kills_clan_tag_to_filteron
else:
    kills_default_clan_tag = None
kills_player_order_continuous = True
kills_rank_strip_start = 60
kills_rank_strip_end = 160
kills_pixels_to_remove = 323  #### No. of pixels to remove from left side of img
#kills_chunk_height = 15650 / kills_num_players ### hunk_height (int): Vertical size of each chunk in pixels
kills_chunk_height_reduction_factor = 0.0 # amount to reduce total chunk height by, chunk_height_reduction_factor * 0.5 from top and bottom of chunk
kills_export_server_data_flag = False
kills_export_clan_data_flag = True
kills_report_type = "kills"

####################################################
############## Donations ###########################
####################################################
donations_flag = False # If TRUE, run analysis on donations
donations_metric = "Donations"
donations_file_name = "donations_10-19-25.JPEG"
donations_server = 870
donations_enemy_server = None
donations_num_players = int(100)
donations_clan_tag_location = None # options: "below", "next_to", None
donations_clan_tag_to_filteron = "GODS"
if donations_clan_tag_location == None:
    donations_default_clan_tag = donations_clan_tag_to_filteron
else:
    donations_default_clan_tag = None
donations_player_order_continuous = True
donations_rank_strip_start = 60
donations_rank_strip_end = 160
donations_pixels_to_remove = 430  #### No. of pixels to remove from left side of img
#donations_chunk_height = 15650 / donations_num_players ### hunk_height (int): Vertical size of each chunk in pixels
donations_chunk_height_reduction_factor = 0.0 # amount to reduce total chunk height by, chunk_height_reduction_factor * 0.5 from top and bottom of chunk
donations_export_server_data_flag = False
donations_export_clan_data_flag = True
donations_report_type = "donations"

####################################################
############## VS ##################################
####################################################
vs_flag = False # If TRUE, run analysis on VS
vs_metric = "VS"
vs_file_name = "vs_10-18-25.JPEG"
vs_server = 870
vs_enemy_server = None
vs_num_players = int(100)
vs_clan_tag_location = "below" # options: "below", "next_to", None
vs_clan_tag_to_filteron = "GODS"
if vs_clan_tag_location == None:
    vs_default_clan_tag = vs_clan_tag_to_filteron
else:
    vs_default_clan_tag = None
vs_player_order_continuous = True
vs_rank_strip_start = 60
vs_rank_strip_end = 160
vs_pixels_to_remove = 380  #### No. of pixels to remove from left side of imgsd
#vs_chunk_height = 156 ### hunk_height (int): Vertical size of each chunk in pixels
vs_chunk_height_reduction_factor = 0.0 # amount to reduce total chunk height by, chunk_height_reduction_factor * 0.5 from top and bottom of chunk
vs_export_server_data_flag = False
vs_export_clan_data_flag = True
vs_report_type = "vs"

####################################################
############## SERVER WARS #########################
####################################################
serverwars_flag = False # If TRUE, run analysis on serverwars
serverwars_metric = "Server_Wars"
serverwars_file_name = "server_wars_10-18-25.JPEG"
serverwars_server = 870
serverwars_enemy_server = 915
serverwars_num_players = int(100)
serverwars_clan_tag_location = "below" # options: "below", "next_to", None
serverwars_clan_tag_to_filteron = "GODS"
if serverwars_clan_tag_location == None:
    serverwars_default_clan_tag = serverwars_clan_tag_to_filteron
else:
    serverwars_default_clan_tag = None
serverwars_player_order_continuous = True
serverwars_rank_strip_start = 60
serverwars_rank_strip_end = 230
serverwars_pixels_to_remove = 380  #### No. of pixels to remove from left side of imgsd
#serverwars_chunk_height = 156 ### hunk_height (int): Vertical size of each chunk in pixels
serverwars_chunk_height_reduction_factor = 0.0 # amount to reduce total chunk height by, chunk_height_reduction_factor * 0.5 from top and bottom of chunk
serverwars_export_server_data_flag = True
serverwars_export_clan_data_flag = False
serverwars_report_type = "server_wars"

class Globals:

    DEFAULTS = {
        "min_score_filter": None,
        "metric_name": None,
        "file_name": None,
        "server": None,
        "enemy_server": None,
        "file_path": None,
        "num_players": None,
        "clan_tag_location": None,
        "clan_tag_to_filteron": None,
        "default_clan_tag": None,
        "player_order_continuous": False,
        "rank_strip_start": None,
        "rank_strip_end": None,
        "pixels_to_remove": None,
        "chunk_height_reduction_factor": None,
        "data_full_time_stamp": None,
        "data_date": None,
        "data_time": None,
        "data_time_zone": None,
        "export_server_data_flag": False,
        "export_clan_data_flag": False,
        "numbers_filtered": None,
        "report_type": None
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
    
    # Build file paths relative to script
    ALLIANCE_REFERENCE_LIST_NAME = "alliance_reference_list.csv"
    ALLIANCE_REFERENCE_LIST_PATH = SCRIPT_DIR / ALLIANCE_REFERENCE_LIST_NAME
    USERNAME_REFERENCE_LIST_NAME = "username_reference_list.csv"
    USERNAME_REFERENCE_LIST_PATH = SCRIPT_DIR / USERNAME_REFERENCE_LIST_NAME
    
    ORIGINAL_IMG_DIRECTORY = "Original_Images"
    ORIGINAL_IMG_DIRECTORY_PATH = SCRIPT_DIR / ORIGINAL_IMG_DIRECTORY
    
    OUTPUT_DIRECTORY = "Processed_Data"
    OUTPUT_DIRECTORY_PATH = SCRIPT_DIR / OUTPUT_DIRECTORY
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
    

def get_file_timestamp(file_path):
    path = Path(file_path)
    ts = path.stat().st_mtime  # last modified time
    local_zone = ZoneInfo("America/New_York")  # or detect dynamically
    dt = datetime.fromtimestamp(ts, tz=local_zone)
    return dt


def crop_left_side(image_path, pixels_to_crop):
    """
    Crops the left side of an image by a specified number of pixels.

    Args:
        image_path (str): The path to the input image.
        pixels_to_crop (int): The number of pixels to remove from the left side.

    Returns:
        PIL.Image.Image: The cropped image object.
    """
    
    
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return None

    width, height = img.size

    # Define the cropping box: (left, upper, right, lower)
    # To crop the left side, the 'left' coordinate starts at 'pixels_to_crop'
    # The 'upper' coordinate remains 0 (top of the image)
    # The 'right' coordinate remains 'width' (right edge of the image)
    # The 'lower' coordinate remains 'height' (bottom of the image)
    crop_box = (pixels_to_crop, 0, width, height)

    cropped_img = img.crop(crop_box)
    return cropped_img


def img_preprocess(pil_img):
    """Convert to grayscale, resize, and threshold."""
    img_cv = np.array(pil_img)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary


def ranks_img_preprocess(image_path):
    """
    Preprocess a full ranks image for OCR (grayscale, threshold, upscale, denoise).
    Returns a preprocessed OpenCV image (numpy array).
    """
    # Load as OpenCV image
    pil_img = image_path.convert("RGB")
    open_cv_image = np.array(pil_img)[:, :, ::-1].copy()

    # Convert to grayscale
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

    # Global Otsu threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Upscale (helps OCR)
    resized = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Denoise
    denoised = cv2.medianBlur(resized, 3)

    return denoised


def match_names(ocr_names, available_names):
    matches, scores, used_names = [], [], []
    for name in ocr_names:
        if not available_names:
            matches.append(None)
            scores.append(0)
            continue
        if name:
            match, score, _ = process.extractOne(name, available_names, scorer=fuzz.WRatio)
            matches.append(match)
            scores.append(score)
            if match:
                used_names.append(match)
                available_names.remove(match)
        else:
            matches.append(None)
            scores.append(0)
            continue
    return matches, scores, used_names

def optimize_names_with_ref_names(df, ocr_names, matches, scores, reference_usernames):
    
    # === OPTIMIZATION LOOP (same as before, applied only to df_gods) ===
    threshold = Globals.OPTIMIZATION_THRESHOLD
    max_attempts_without_improvement = Globals.MAX_OPTI_STEPS_WITHOUT_IMPROVEMENT
    attempts = 0
    best_total_score = sum(scores)
    while attempts < max_attempts_without_improvement:
        low_score_indices = [i for i, s in enumerate(scores) if s < threshold]
        if not low_score_indices:
            break
        random.shuffle(low_score_indices)
        remaining_usernames = reference_usernames.copy()
        for i, match in enumerate(matches):
            if i not in low_score_indices and match in remaining_usernames:
                remaining_usernames.remove(match)
        temp_matches, temp_scores = matches.copy(), scores.copy()
        for i in low_score_indices:
            ocr_name = ocr_names[i]
            if not remaining_usernames or not ocr_name:
                temp_matches[i] = None
                temp_scores[i] = 0
                continue
            new_match, new_score, _ = process.extractOne(ocr_name, remaining_usernames, scorer=fuzz.WRatio)
            temp_matches[i] = new_match
            temp_scores[i] = new_score
            if new_match:
                remaining_usernames.remove(new_match)
        new_total_score = sum(temp_scores)
        if new_total_score > best_total_score:
            print(f"Improved total score: {new_total_score} (was {best_total_score})")
            matches, scores = temp_matches, temp_scores
            best_total_score = new_total_score
            df['Matched_Name'] = matches
            df['Match_Score'] = scores
            attempts = 0
        else:
            attempts += 1
            
    print(f"Final total match score: {best_total_score}")        
    return df


def extract_rank_with_positions(image_path, chunk_height, chunk_height_reduction_factor, continuous_scores=False):
    """Extract ranks along with their vertical positions."""
    
    #print(f"Extracting ranks")
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789 '
    custom_config_2nd_try = r'--oem 3 --psm 6'
    img = Image.open(image_path)
    width, height = img.size
    #results_i = []
    results = []
    reduced_chunk_height_top = int(np.round(chunk_height * chunk_height_reduction_factor * 0.5,0))
    reduced_chunk_height_bottom = int(np.round(chunk_height * chunk_height_reduction_factor * 0.5,0))

    record_num = 1
    
    num_chunks = int(height / chunk_height)
    for i in range(0, num_chunks):
        y = i * chunk_height
        
        if y + chunk_height // 2 > height:
            #print(f"Completed extracting ranks")
            return results
        
        if continuous_scores == True:
            results.append({"rank": record_num, "y": int(y + chunk_height // 2)})
        
        else:
            # if record_num == 101:
            #     print("pause")
            #print(record_num)
            y1 = y + reduced_chunk_height_top
            y2 = y + chunk_height - reduced_chunk_height_bottom
            box = (0, y1, width, min(y2, height))
            chunk = img.crop(box)
            preprocessed = ranks_img_preprocess(chunk)
            data = pytesseract.image_to_data(preprocessed, config=custom_config, output_type=pytesseract.Output.DICT)
            #data["text"] = "".join(data["text"])
            
            results_i = []
            for i, text in enumerate(data["text"]):
                if text.strip().isdigit():
                    results_i.append({"rank": int(text.strip()), "y": int(y1 + data["top"][i])})
                # else:
                #     results_i.append({"rank": None, "y": y + data["top"][i]})
            
            # if continuous_scores == True:
            #     results.append({"rank": record_num, "y": int(y + chunk_height // 2)})
            if results_i:
                if record_num in (1, 2, 3):
                    if results_i[0]["rank"] == record_num:
                        results.append(results_i[0])   # store dict directly
                    else:
                        results.append({"rank": None, "y": int(y + chunk_height // 2)})
                else:
                    results.append(results_i[0])
            else:
                results.append({"rank": None, "y": int(y + chunk_height // 2)})

        record_num += 1
            
    #print(f"Completed extracting ranks")
    return results

def extract_text_with_positions(image_path, chunk_height):
    """
    Extract exactly one line of text per chunk_height slice.
    Returns list of {text, y}.
    """
    #print(f"Extracting text")
    #custom_config = r'--oem 3 --psm 6'
    #custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789[]'
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789[], '

    img = Image.open(image_path)
    width, height = img.size
    results = []
    results_data = []
    results_chunks = []
    img_position_box = []
    
    num_chunks = int(height / chunk_height)
    for i in range(0, num_chunks):
        y = i * chunk_height
        if y + chunk_height // 2 > height:
            #print(f"Completed extracting text")
            return results
        else:
            box = (0, y, width, min(y + chunk_height, height))
            img_position_box.append(box)
            chunk = img.crop(box)
            preprocessed = img_preprocess(chunk)
    
            # OCR the entire chunk as one line
            text_data_df = pytesseract.image_to_data(preprocessed, config=custom_config, output_type=pytesseract.Output.DATAFRAME)

            # Filter (word-level, valid confidence, non-empty text)
            text_data_df = text_data_df[(text_data_df['conf'] > -1)]
            text_data_df.reset_index(drop=True)
            results_data.append(text_data_df)
            results_chunks.append(chunk)
            # Convert back to dict-of-lists
            text_data = text_data_df.to_dict(orient="list")
            text = text_data["text"]
            
            
            # text_data = pytesseract.image_to_data(preprocessed, config=custom_config, output_type=pytesseract.Output.DICT)
            # text_data_df = pytesseract.image_to_data(preprocessed, config=custom_config, output_type=pytesseract.Output.DATAFRAME)
            # text_data_cleaned = text_data_df[(text_data_df['conf'] > -1)]

            
            # text = pytesseract.image_to_string(preprocessed, config=custom_config).strip()
            # text = data[(data['level'] == 5) & (data['conf'] > -1)]
            
            if text:
                results.append({
                    "text": text,
                    "y": y + chunk_height // 2   # middle of the chunk as reference position
                })
            else:
                results.append({
                    "text": None,
                    "y": y + chunk_height // 2   # middle of the chunk as reference position
                })
                
            
    #print(f"Completed extracting text")
    return results, results_data, results_chunks, img_position_box

def parse_block(raw_block, raw_block_data, clan_tag_location, default_clan_tag=None):
    """
    Parse a VS-style OCR block:
    [TAG] Username
    123,456,789
    (maybe extra junk)
    
    Returns dict: {clan_tag, username, score}
    """
    #lines = [l.strip() for l in raw_block.splitlines() if l.strip()]
    lines = raw_block
    
    clan_tag, username, score = None, None, None
    

    # Extract numbers >= 10,000,000 in one shot
    numbers_filtered = raw_block_data["text"].str.extract(Globals.variables_state["numbers_filtered"])[0]
    
    #numbers_filtered = raw_block_data["text"].str.extract(r"(\d{1,3}(?:,\d{3})+|\d+)")[0]

    
    # Convert to int safely
    numbers_int = pd.to_numeric(numbers_filtered.str.replace(",", ""), errors="coerce")

    
    # Step 2: Filter for > 10,000,000
    #nonlocal min_score_filter # pulls value from current min_score_filter set in ___main___
    min_score_filter = Globals.variables_state["min_score_filter"]
    mask = numbers_int > min_score_filter
    
    # Step 3: Get indices where condition is True
    score_index = numbers_int[mask].index.tolist()
    try:
        score_index = int(score_index[0])
        score = numbers_int[score_index]
    except:
        score_index = None
        score = None
    
    if clan_tag_location == "next_to":
        
        if len(raw_block_data["text"]) == 1 and score is not None:
            combined_text = re.sub(numbers_filtered[score_index], "", raw_block_data.loc[0,"text"])
            tag_match = re.match(r"\[(.*?)\]\s*(.*)", combined_text)
            tag_match_dropped_left_bracket = re.match(r"(.*?)\]\s*(.*)", combined_text)
            
            if tag_match:
                clan_tag = tag_match.group(1).strip()
                username = tag_match.group(2).strip()
                
            elif tag_match_dropped_left_bracket:
                clan_tag = tag_match_dropped_left_bracket.group(1).strip()
                username = tag_match_dropped_left_bracket.group(2).strip()
        
        
        if score_index and score_index == 1:
            tag = re.sub(" ", "", raw_block_data.loc[0,"text"])
            tag = re.sub(",", "", tag)
            tag_match = re.match(r"\[(.*?)\]\s*(.*)", tag)
            tag_match_dropped_left_bracket = re.match(r"(.*?)\]\s*(.*)", tag)
            
            if tag_match:
                clan_tag = tag_match.group(1).strip()
                username = tag_match.group(2).strip()
                
            elif tag_match_dropped_left_bracket:
                clan_tag = tag_match_dropped_left_bracket.group(1).strip()
                username = tag_match_dropped_left_bracket.group(2).strip()
                
            else:
                clan_tag = None
                username = raw_block_data.loc[0,"text"]
                
            
        if score_index and score_index > 1:
            tag_combined = " ".join(raw_block_data.loc[:score_index-1, "text"].astype(str))
            tag_combined = re.sub(" ","",tag_combined)
            tag_combined = re.sub(",","",tag_combined)
            
            tag_match = re.match(r"\[(.*?)\]\s*(.*)", tag_combined)
            tag_match_dropped_left_bracket = re.match(r"(.*?)\]\s*(.*)", tag_combined)
            
            if tag_match:
                clan_tag = tag_match.group(1).strip()
                username = tag_match.group(2).strip()
                
            elif tag_match_dropped_left_bracket:
                clan_tag = tag_match_dropped_left_bracket.group(1).strip()
                username = tag_match_dropped_left_bracket.group(2).strip()
            
            else:
                clan_tag = raw_block_data.loc[0,"text"]
                username = re.sub(clan_tag,"",tag_combined)
                username = re.sub(" ","",username)  
                
        if score_index == None or score == None:
            tag_combined = " ".join(raw_block_data.loc[:, "text"].astype(str))
            tag_combined = re.sub(" ","",tag_combined)
            tag_combined = re.sub(",","",tag_combined)
            
            tag_match = re.match(r"\[(.*?)\]\s*(.*)", tag_combined)
            tag_match_dropped_left_bracket = re.match(r"(.*?)\]\s*(.*)", tag_combined)
            
            if tag_match:
                clan_tag = tag_match.group(1).strip()
                username = tag_match.group(2).strip()
                
            elif tag_match_dropped_left_bracket:
                clan_tag = tag_match_dropped_left_bracket.group(1).strip()
                username = tag_match_dropped_left_bracket.group(2).strip()
            
            else:
                clan_tag = None
                username = raw_block_data.loc[0,"text"]
                username = re.sub(" ","",username)  
                
    if clan_tag_location == "below":
        
        if score_index == 1:
            username = re.sub(" ", "", raw_block_data.loc[0,"text"])
            username = re.sub(",", "", username)

        if score_index > 1:
            username = " ".join(raw_block_data.loc[:score_index-1, "text"].astype(str))
            username = re.sub(" ","",username)
            username = re.sub(",","",username)
        
        tag_combined = " ".join(raw_block_data.loc[score_index+1:, "text"].astype(str))
        tag_match = re.match(r"\[(.*?)\]\s*(.*)", tag_combined)
        tag_match_dropped_left_bracket = re.match(r"(.*?)\]\s*(.*)", tag_combined)

        if tag_match:
            clan_tag = tag_match.group(1).strip()
            
        elif tag_match_dropped_left_bracket:
            clan_tag = tag_match_dropped_left_bracket.group(1).strip()
            
    if clan_tag_location == None:
        
        if score_index == 1:
            username = re.sub(" ", "", raw_block_data.loc[0,"text"])
            username = re.sub(",", "", username)

        if score_index > 1:
            username = " ".join(raw_block_data.loc[:score_index-1, "text"].astype(str))
            username = re.sub(" ","",username)
            username = re.sub(",","",username)
        
        clan_tag = default_clan_tag

    
    print({
        "clan_tag": clan_tag,
        "username": username,
        "score": score
    })
    
    return {
        "clan_tag": clan_tag,
        "username": username,
        "score": score
    }



def best_fit_ranks(rank_results, expected_count):
    """
    Force ranks into 1..expected_count by best alignment with OCR results.
    """
    if not rank_results:
        return list(range(1, expected_count + 1)), [None] * expected_count

    #rank_results_sorted = sorted(rank_results, key=lambda r: r["y"])
    rank_results_sorted = rank_results
    ocr_ranks = [r["rank"] for r in rank_results_sorted]
    y_positions = [r["y"] for r in rank_results_sorted]

    # Stretch/compress OCR sequence into 1..expected_count
    fitted_ranks = list(range(1, expected_count + 1))
    fitted_y = np.linspace(min(y_positions), max(y_positions), expected_count)

    return fitted_ranks, fitted_y


def get_data():

    print(f"Starting extracting text from {Globals.variables_state["metric_name"]} scrolling screenshot...")

    metric_name = Globals.variables_state["metric_name"]
    rank_strip_start = Globals.variables_state["rank_strip_start"]
    rank_strip_end = Globals.variables_state["rank_strip_end"]
    num_players = Globals.variables_state["num_players"]
    chunk_height_reduction_factor = Globals.variables_state["chunk_height_reduction_factor"]
    player_order_continuous = Globals.variables_state["player_order_continuous"]
    pixels_to_remove = Globals.variables_state["pixels_to_remove"]
    clan_tag_location = Globals.variables_state["clan_tag_location"]
    clan_tag_to_filteron = Globals.variables_state["clan_tag_to_filteron"]
    default_clan_tag = Globals.variables_state["default_clan_tag"]
    server = Globals.variables_state["server"]
    enemy_server = Globals.variables_state["enemy_server"]
    report_type = Globals.variables_state["report_type"]
    
    
    file_path = Globals.ORIGINAL_IMG_DIRECTORY_PATH / report_type / Globals.variables_state["file_name"]

    dt = get_file_timestamp(file_path)
    time_stamp = dt.strftime("%Y-%m-%d_%I-%M%p_%Z")
    date = dt.strftime("%Y-%m-%d")
    time = dt.strftime("%I:%M%p")
    time_zone = dt.strftime("%Z")
    
    Globals.variables_state["file_path"] = file_path
    Globals.variables_state["data_full_time_stamp"] = time_stamp
    Globals.variables_state["data_date"] = date
    Globals.variables_state["data_time"] = time
    Globals.variables_state["data_time_zone"] = time_zone

    # === PASS 1: Extract ranks with y positions ===
    print("Extracting ranks from image")
    rank_crop_box = (rank_strip_start, 0, rank_strip_end, Image.open(file_path).size[1])
    rank_img = Image.open(file_path).crop(rank_crop_box)
    # Calculate expected entries dynamically
    img_height = rank_img.size[1]
    chunk_height = float(img_height / num_players)
    
    # store calculated variables
    Globals.variables_state["chunk_height"] = chunk_height

    #expected_count = round(img_height / vs_chunk_height)
    #print(f"Detected ~{expected_count} entries from image height {img_height}px / chunk {vs_chunk_height}px")
    rank_img_path = Globals.PROCESSED_IMG_OUTPUT_DIRECTORY_PATH / f"{metric_name}_{time_stamp}_ranks.jpg"
    rank_img.save(rank_img_path)

    rank_results = extract_rank_with_positions(rank_img_path, chunk_height, chunk_height_reduction_factor, player_order_continuous)
    #fitted_ranks, fitted_y = best_fit_ranks(rank_results, expected_count)  # VS list = 100
    ranks_values = [r["rank"] for r in rank_results]
    ranks_yposition = [r["y"] for r in rank_results]
    print("Completed extracting ranks from image")
    
    print(f"Extracting {metric_name} data on server {server} from image")
    # === PASS 2: Extract usernames+scores with y positions ===
    main_crop = crop_left_side(file_path, pixels_to_remove)
    main_img_path = Globals.PROCESSED_IMG_OUTPUT_DIRECTORY_PATH / f"{metric_name}_{time_stamp}_cropped.jpg"
    main_crop.save(main_img_path)

    text_results, text_results_data, results_chunks, img_position_box = extract_text_with_positions(main_img_path, chunk_height)

    extracted_names, extracted_numbers, extracted_clan,  = [], [], []
    for item, item_data, item_rank, item_chunk, item_img_position_box in zip(text_results, text_results_data, ranks_values, results_chunks, img_position_box):
        try:
            
            results = parse_block(item["text"], item_data.reset_index(drop=True), clan_tag_location, default_clan_tag)
            name, score, clan_tag = fill_blank_extracted_names_and_scores_v2(results['username'], results['score'], results['clan_tag'], item_rank, item_chunk, item_img_position_box)                

            extracted_names.append(name)
            extracted_numbers.append(score)
            extracted_clan.append(clan_tag)
            #fill_blank_extracted_names_and_scores_v2(results['username'], results['score'], results['clan_tag'])                
                
        except:
            try:
                name, score, clan_tag = fill_blank_extracted_names_and_scores_v2(None, None, None, item_rank, item_chunk, item_img_position_box)                
            
                extracted_names.append(name)
                extracted_numbers.append(score)
                extracted_clan.append(clan_tag)
                
            except:
                extracted_names.append(None)
                extracted_numbers.append(np.nan)
                extracted_clan.append(None)

    # === ALIGN RANKS AND BUILD DF ===
    df_all = pd.DataFrame({
        "Date": date,
        "Time": time,
        "Time_Zone": time_zone,
        "Server": server,
        "Enemy_Server": enemy_server,
        "Overall_Rank": ranks_values[:len(extracted_names)],
        "Clan_Tag": extracted_clan,
        "Extracted_Name": extracted_names,
        "Matched_Name": None,
        "Match_Score": None,
        "Score": extracted_numbers,
        "Report_Type": report_type
        
    })
    
    #df_all = fill_blank_extracted_names_and_scores(df_all)
    
    print(f"Extracted all {metric_name} data on server {server}")

    # === LOAD REFERENCE LISTS ===
    alliance_df = pd.read_csv(Globals.ALLIANCE_REFERENCE_LIST_PATH)
    valid_alliances = alliance_df['Name'].tolist()

    username_df = pd.read_csv(Globals.USERNAME_REFERENCE_LIST_PATH)
    reference_usernames = username_df['Name'].tolist()


    # === Filter Clan only ===
    print(f"Filtering {metric_name} data for {clan_tag_to_filteron} and matching usernames")
    df_clan = df_all[df_all['Clan_Tag'] == clan_tag_to_filteron].copy()
        # === Initial matching ===
    available_usernames = reference_usernames.copy()
    ocr_names = df_clan['Extracted_Name'].tolist()

    matches, scores, used_names = match_names(ocr_names, available_usernames.copy())
    _best_total_score = sum(scores)

    df_clan['Matched_Name'] = matches
    df_clan['Match_Score'] = scores

    print(f"Initial total match score: {_best_total_score}")
    df_clan = optimize_names_with_ref_names(df_clan, ocr_names, matches, scores, reference_usernames)

    # === Rescue Step: check unmatched names in df_all ===
    unmatched = df_all[df_all['Clan_Tag'].isna()].copy()
    if not unmatched.empty:
        rescue_matches, rescue_scores, _ = match_names(
            unmatched['Extracted_Name'].tolist(),
            reference_usernames
        )
        unmatched['Matched_Name'] = rescue_matches
        unmatched['Match_Score'] = rescue_scores

        # Only keep strong matches (e.g., score >= 80) and mark them as clan_tag_to_filteron
        rescue_confirmed = unmatched[unmatched['Match_Score'] >= 80].copy()
        rescue_confirmed['Clan_Tag'] = clan_tag_to_filteron

        if not rescue_confirmed.empty:
            print(f"\nRescued {len(rescue_confirmed)} {clan_tag_to_filteron} players with missing clan tag")
            # Add them to df_gods
            df_clan = pd.concat([df_clan, rescue_confirmed], ignore_index=True)

    # === Finalize Output ===
    desired_order = ['Date','Time','Time_Zone','Server','Enemy_Server','Overall_Rank','Clan_Tag',
                          'Extracted_Name','Matched_Name','Match_Score',"Score","Report_Type"]
    df_clan = df_clan[desired_order]
    print(f"Completed filtering {metric_name} data for {clan_tag_to_filteron} and matching usernames")
    
    
    # Update df1_indexed with values from df2_indexed
    df_all.update(df_clan)


    df_all = df_all[desired_order]



    return df_all, df_clan

    
def manual_confirmation_of_low_scores(df):    
    # === MANUAL CONFIRMATION FOR LOW SCORES ===
    interactive_threshold = Globals.INTERACTIVE_OPTIMIZATION_THRESHOLD
    
    metric_name = Globals.variables_state["metric_name"]

    for idx, row in df.iterrows():
        score = row['Match_Score']
        metric_value = row['Score']
        if score is None or score >= interactive_threshold:
            continue  # Skip confident matches

        ocr_name = row['Extracted_Name']
        suggested_match = row['Matched_Name']
    
        print("\n------------------------------------")
        print(f"OCR Name        : {ocr_name}")
        print(f"Suggested Match : {suggested_match}")
        print(f"Match Score     : {score}")
        print(f"{metric_name} Score: {metric_value}")
        response = input("Accept match? [Y/n/edit]: ").strip().lower()
    
        if response == 'n':
            df.at[idx, 'Matched_Name'] = None
            df.at[idx, 'Match_Score'] = 0
        elif response == 'edit':
            new_name = input("Enter correct username: ").strip()
            df.at[idx, 'Matched_Name'] = new_name
            df.at[idx, 'Match_Score'] = 100  # Considered confirmed
        else:
            # Accept as is
            pass
    
    return df


def save_with_prompt(df_all, df_clan):

    def wait_for_file_close(file_path, timeout=30, check_interval=2):
        """Wait until file is not locked by another process (Excel, etc.)."""
        start = time.time()
        while True:
            try:
                with open(file_path, "a"):
                    return True  # File can be opened → it's free
            except PermissionError:
                if time.time() - start > timeout:
                    print(f"⏳ Timeout: {file_path} is still open. Skipping save.")
                    return False
                print(f"⚠️ {file_path} is open in another program. Waiting...")
                time.sleep(check_interval)

    def execute_save(df, file_path):
        file_path = Path(file_path)

        # Check if already exists
        if file_path.exists():
            choice = input(f"⚠️ File {file_path} already exists. Overwrite? (y/n): ").strip().lower()
            if choice != "y":
                print("❌ Skipped saving.")
                return

            # Wait for file to be closed
            if not wait_for_file_close(file_path):
                return  # Timed out, skip save

        # Try saving
        try:
            df.to_csv(file_path, index=False)
            print(f"✅ Saved {file_path}")
        except Exception as e:
            print(f"❌ Could not save {file_path}: {e}")

    # Pull config values
    metric_name = Globals.variables_state["metric_name"]
    export_server_data_flag = Globals.variables_state["export_server_data_flag"]
    export_clan_data_flag = Globals.variables_state["export_clan_data_flag"]
    time_stamp = Globals.variables_state["data_full_time_stamp"]
    clan_tag_to_filteron = Globals.variables_state["clan_tag_to_filteron"]
    server = Globals.variables_state["server"]
    enemy_server = Globals.variables_state["enemy_server"]
    

    # Save full server data
    
    if metric_name == 'Enemy_Total_Hero_Power':
        server_name_for_save_file = enemy_server
    elif metric_name == 'Server_Wars':
        server_name_for_save_file = f'{server}_vs_{enemy_server}'
    else:
        server_name_for_save_file = server
        
    if export_server_data_flag and df_all is not None and not df_all.empty:
        output_csv_all = Globals.OUTPUT_DIRECTORY_PATH / f"LastWar_{metric_name}_ALL{server_name_for_save_file}_{time_stamp}.csv"
        execute_save(df_all, output_csv_all)

    # Save clan-only data
    if export_clan_data_flag and df_clan is not None and not df_clan.empty:
        output_csv_clan = Globals.OUTPUT_DIRECTORY_PATH / f"LastWar_{metric_name}_{clan_tag_to_filteron}only_{time_stamp}.csv"
        execute_save(df_clan, output_csv_clan)

def fill_blank_extracted_names_and_scores_v2(name, metric_value, clan_tag, rank, img_chunk, img_position_box):

    metric_name = Globals.variables_state["metric_name"]

    
    if not name or not clan_tag or not metric_value or np.isnan(metric_value):
        
        def create_gui(metric_name, prompt_text, img_chunk, name, metric_value, clan_tag, img_position_box):
            
            if not clan_tag:
                clan_tag = "No_Clan"
                
            root = tk.Tk()
            root.title("Fill blank names, scores, and clan tags")

            # 1. Displaying Text/Data
            data_label = tk.Label(root, text=f"Please confirm Clan Tag, Name, and {metric_name} Score.", font=("Arial", 12))
            data_label.pack(pady=5)
            user_input_name = name
            user_input_metric_value = metric_value
            user_input_clan_tag = clan_tag
            
            # 2. Displaying an Image
            try:
                # Replace 'your_image.png' with the actual path to your image file
                # image_path = "your_image.png" 
                img = Image.open(Globals.variables_state["file_path"])
                img_original_width, img_original_height = img.size
                original_chunk_width, original_chunk_height = img_chunk.size
                crop_box = (img_position_box[0], int(img_position_box[1]), img_original_width, int(img_position_box[3]))
                img_cropped = img.crop(crop_box)
                
                new_height = 100 # px
                
                aspect_ratio = img_original_width / original_chunk_height
            
                new_width = int(new_height * aspect_ratio)
                img = img_cropped.resize((new_width, new_height), Image.LANCZOS) # Use LANCZOS for high-quality downscaling
                
                #img = img_chunk.resize((200, 200), Image.Resampling.LANCZOS) # Resize as needed
                #img = img_chunk
                photo_img = ImageTk.PhotoImage(img)

                image_label = tk.Label(root, image=photo_img)
                image_label.image = photo_img # Keep a reference
                image_label.pack(pady=5)

            except Exception as e:
                error_label = tk.Label(root, text=f"Error loading image: {e}")
                error_label.pack(pady=10)

            # 3. Asking for Input
            input_label = tk.Label(root, text=prompt_text, justify="left", font=("Arial", 12))
            input_label.pack(pady=5)

            # entry_field_clan_tag = tk.Entry(root, width=30)
            # entry_field_clan_tag.insert(0, str(clan_tag))
            # entry_field_clan_tag.pack(pady=10)


            # Create a Frame to group the Label and Entry
            frame_clan_tag = tk.Frame(root)
            frame_clan_tag.pack(padx=10, pady=5)
            frame_name = tk.Frame(root)
            frame_name.pack(padx=10, pady=5)
            frame_metric_value = tk.Frame(root)
            frame_metric_value.pack(padx=10, pady=5)
            
            
            # Create a Label widget and pack it to the left within the frame
            label_clan_tag = tk.Label(frame_clan_tag, text="Current Clan Tag:", font=("Arial", 12))
            label_clan_tag.pack(side=tk.LEFT, padx=5)
            label_name = tk.Label(frame_name, text="Current Name:", font=("Arial", 12))
            label_name.pack(side=tk.LEFT, padx=5)
            label_metric_value = tk.Label(frame_metric_value, text=f"Current {metric_name} Score:", font=("Arial", 12))
            label_metric_value.pack(side=tk.LEFT, padx=5)
            
            # Create an Entry widget and pack it to the left within the frame
            entry_field_clan_tag = tk.Entry(frame_clan_tag, width=30, font=("Arial", 12))
            entry_field_clan_tag.insert(0, str(clan_tag))
            entry_field_clan_tag.pack(side=tk.LEFT, padx=5)
            
            entry_field_name = tk.Entry(frame_name, width=30, font=("Arial", 12))
            entry_field_name.insert(0, str(name))
            entry_field_name.pack(pady=5)
            
            entry_field_metric_value = tk.Entry(frame_metric_value, width=30, font=("Arial", 12))
            entry_field_metric_value.insert(0, str(metric_value))
            entry_field_metric_value.pack(pady=5)




            def submit_and_close():
                nonlocal user_input_name, user_input_metric_value, user_input_clan_tag
                user_input_name = entry_field_name.get()
                user_input_metric_value = entry_field_metric_value.get()
                user_input_clan_tag = entry_field_clan_tag.get()
                root.destroy()

            submit_button = tk.Button(root, text="Submit", command=submit_and_close)
            submit_button.pack(pady=10)

            feedback_label = tk.Label(root, text="")
            feedback_label.pack(pady=5)

            root.mainloop()
            
            return user_input_name, user_input_metric_value, user_input_clan_tag

        prompt_text = (f"""
                     
                       Possible Overall Rank:             {rank}
                       
                       Clan Tag:                          {clan_tag}
                       
                       Extracted_Name:                    {name}
                       
                       {metric_name} Score:               {metric_value}
                       
                       """)
                       
        #response = input("Actual Name? [type name or press enter to skip]: ").strip()
        
        name, metric_value, clan_tag = create_gui(metric_name, prompt_text, img_chunk, name, metric_value, clan_tag, img_position_box)    
        
        # remove non number or decimal point characters in user_input_metric_value
        metric_value = re.sub(r'[^0-9.]', '', metric_value)
    #     if not response:
    #         # Accept as is
    #         pass
        
    #     else:
    #         name = response

    # if not clan_tag:
    #     item_chunk.show()
    #     print("\n------------------------------------")
    #     print("No Clan tag in record.")
    #     print(f"Possible Overall Rank    : {rank}")
    #     print(f"Clan Tag        : {clan_tag}")
    #     print(f"Extracted_Name  : {name}")
    #     print(f"{metric_name} Score: {metric_value}")
    #     response = input("Is this player in a clan? [y, n, or press enter to skip]: ").strip()
    
    #     if response == "y":
    #         response = input("Enter Clan Tag or press enter to skip: ").strip()
    #         clan_tag = response

    #     elif response == "n":
    #         clan_tag = "No_Clan"
        
    #     elif not response:
    #         # Accept as is
    #         pass
            
    # if not metric_value or np.isnan(metric_value):
    #     item_chunk.show()
    #     print("\n------------------------------------")
    #     print("No score in record.")
    #     print(f"Possible Overall Rank    : {rank}")
    #     print(f"Clan Tag        : {clan_tag}")
    #     print(f"Extracted_Name  : {name}")
    #     print(f"{metric_name} Score: {metric_value}")
    #     response = input("Actual score? [type score or press enter to skip]: ").strip()
    
    #     if not response:
    #         # Accept as is
    #         pass
        
    #     else:
    #         metric_value = response
        
    return name, metric_value, clan_tag
        
def fill_blank_extracted_names_and_scores(df):
    df_blank_names = df.loc[df["Extracted_Name"].isnull() | df["Score"].isnull()]
    metric_name = Globals.variables_state["metric_name"]
    

    for idx, row in df_blank_names.iterrows():
        clan = row['Clan_Tag']
        name = row['Extracted_Name']
        rank = row['Overall_Rank']
        metric_value = row['Score']

        print("\n------------------------------------")
        print(f"Overall Rank    : {rank}")
        print(f"Clan Tag        : {clan}")
        print(f"Extracted_Name  : {name}")
        print(f"{metric_name} Score: {metric_value}")
        response = input("Actual Name? [type name or press enter to skip]: ").strip()
    
        if not response:
            # Accept as is
            pass
        
        else:
            new_name = response
            df.at[idx, 'Extracted_Name'] = new_name
    
        if not clan:
            print("\n------------------------------------")
            print("No Clan tag in record.")
            print(f"Overall Rank    : {rank}")
            print(f"Clan Tag        : {clan}")
            print(f"Extracted_Name  : {new_name}")
            print(f"{metric_name} Score: {metric_value}")
            response = input("Actual Clan Tag? [type Clan Tag or press enter to skip]: ").strip()
        
            if not response:
                # Accept as is
                pass
            
            else:
                new_clan_tag = response
                df.at[idx, 'Clan_Tag'] = new_clan_tag
                
        if not metric_value or np.isnan(metric_value):
            print("\n------------------------------------")
            print("No score in record.")
            print(f"Overall Rank    : {rank}")
            print(f"Clan Tag        : {new_clan_tag}")
            print(f"Extracted_Name  : {new_name}")
            print(f"{metric_name} Score: {metric_value}")
            response = input("Actual score? [type score or press enter to skip]: ").strip()
        
            if not response:
                # Accept as is
                pass
            
            else:
                new_score = response
                df.at[idx, 'Score'] = new_score
        
    return df

def get_final_names(df_all, df_clan):
    
    desired_order = ['Date','Time','Time_Zone','Server','Enemy_Server','Overall_Rank','Clan_Tag',
                          'Extracted_Name','Matched_Name','Match_Score','Name','Score',"Report_Type"]
    
    df_all['Name'] = np.where(df_all['Matched_Name'].notna(), df_all['Matched_Name'], df_all['Extracted_Name'])
    
    df_all = df_all[desired_order]
    
    if df_clan is not None:
        df_clan['Name'] = np.where(df_clan['Matched_Name'].notna(), df_clan['Matched_Name'], df_clan['Extracted_Name'])
        df_clan = df_clan[desired_order]
        

    return df_all, df_clan

if __name__ == "__main__":


    ### Total Hero Power ###
    if thp_flag == True:
        
        Globals.reset()
        
        Globals.variables_state.update({
            "metric_name": str(thp_metric),
            "file_name": str(thp_file_name),
            "server": thp_server,
            "enemy_server": thp_enemy_server,
            "num_players": int(thp_num_players),
            "clan_tag_location": str(thp_clan_tag_location) if thp_clan_tag_location else None,
            "clan_tag_to_filteron": str(thp_clan_tag_to_filteron) if thp_clan_tag_to_filteron else None,
            "default_clan_tag": str(thp_clan_tag_to_filteron) if not thp_clan_tag_location else None,
            "player_order_continuous": bool(thp_player_order_continuous),
            "rank_strip_start": int(thp_rank_strip_start),
            "rank_strip_end": int(thp_rank_strip_end),
            "pixels_to_remove": int(thp_pixels_to_remove),
            "chunk_height_reduction_factor": float(thp_chunk_height_reduction_factor),
            "export_server_data_flag": bool(thp_export_server_data_flag),
            "export_clan_data_flag" : bool(thp_export_clan_data_flag),
            "min_score_filter": Globals.MIN_SCORE_THP,
            "numbers_filtered": Globals.NUMBERS_FILTERED_1,
            "report_type": thp_report_type
            })

        df_all, df_clan = get_data()
        
        if df_clan is not None:
            df_clan = manual_confirmation_of_low_scores(df_clan)
            df_all.update(df_clan)
            

        df_all, df_clan = get_final_names(df_all, df_clan)
        
        
        # Save CSV
        save_with_prompt(df_all, df_clan)

        
    if enemy_thp_flag == True:
        
        Globals.reset()
        
        Globals.variables_state.update({
            "metric_name": str(enemy_thp_metric),
            "file_name": str(enemy_thp_file_name),
            "server": enemy_thp_server,
            "enemy_server": enemy_thp_enemy_server,
            "num_players": int(enemy_thp_num_players),
            "clan_tag_location": str(enemy_thp_clan_tag_location) if enemy_thp_clan_tag_location else None,
            "clan_tag_to_filteron": str(enemy_thp_clan_tag_to_filteron) if enemy_thp_clan_tag_to_filteron else None,
            "default_clan_tag": str(enemy_thp_clan_tag_to_filteron) if not enemy_thp_clan_tag_location else None,
            "player_order_continuous": bool(enemy_thp_player_order_continuous),
            "rank_strip_start": int(enemy_thp_rank_strip_start),
            "rank_strip_end": int(enemy_thp_rank_strip_end),
            "pixels_to_remove": int(enemy_thp_pixels_to_remove),
            "chunk_height_reduction_factor": float(enemy_thp_chunk_height_reduction_factor),
            "export_server_data_flag": bool(enemy_thp_export_server_data_flag),
            "export_clan_data_flag" : bool(thp_export_clan_data_flag),
            "min_score_filter": Globals.MIN_SCORE_THP,
            "numbers_filtered": Globals.NUMBERS_FILTERED_1,
            "report_type": enemy_thp_report_type
            })

        df_all, df_clan = get_data()
        
        if df_clan is not None:
            df_clan = manual_confirmation_of_low_scores(df_clan)
            df_all.update(df_clan)
            
        df_all, df_clan = get_final_names(df_all, df_clan)
    
        # Save CSV
        save_with_prompt(df_all, df_clan)
    

        
    ### Kills ###
    if kills_flag == True:
        
        Globals.reset()
        
        Globals.variables_state.update({
            "metric_name": str(kills_metric),
            "file_name": str(kills_file_name),
            "server": kills_server,
            "enemy_server": kills_enemy_server,
            "num_players": int(kills_num_players),
            "clan_tag_location": str(kills_clan_tag_location) if kills_clan_tag_location else None,
            "clan_tag_to_filteron": str(kills_clan_tag_to_filteron) if kills_clan_tag_to_filteron else None,
            "default_clan_tag": str(kills_clan_tag_to_filteron) if not kills_clan_tag_location else None,
            "player_order_continuous": bool(kills_player_order_continuous),
            "rank_strip_start": int(kills_rank_strip_start),
            "rank_strip_end": int(kills_rank_strip_end),
            "pixels_to_remove": int(kills_pixels_to_remove),
            "chunk_height_reduction_factor": float(kills_chunk_height_reduction_factor),
            "export_server_data_flag": bool(kills_export_server_data_flag),
            "export_clan_data_flag" : bool(kills_export_clan_data_flag),
            "min_score_filter": Globals.MIN_SCORE_KILLS,
            "numbers_filtered": Globals.NUMBERS_FILTERED_2,
            "report_type": kills_report_type
            })

        df_all, df_clan = get_data()
        
        if df_clan is not None:
            df_clan = manual_confirmation_of_low_scores(df_clan)
            df_all.update(df_clan)
            
        df_all, df_clan = get_final_names(df_all, df_clan)
    
        # Save CSV
        save_with_prompt(df_all, df_clan)


    ### Donations ###
    if donations_flag == True:
        
        Globals.reset()
        
        Globals.variables_state.update({
            "metric_name": str(donations_metric),
            "file_name": str(donations_file_name),
            "server": donations_server,
            "enemy_server": donations_enemy_server,
            "num_players": int(donations_num_players),
            "clan_tag_location": str(donations_clan_tag_location) if donations_clan_tag_location else None,
            "clan_tag_to_filteron": str(donations_clan_tag_to_filteron) if donations_clan_tag_to_filteron else None,
            "default_clan_tag": str(donations_clan_tag_to_filteron) if not donations_clan_tag_location else None,
            "player_order_continuous": bool(donations_player_order_continuous),
            "rank_strip_start": int(donations_rank_strip_start),
            "rank_strip_end": int(donations_rank_strip_end),
            "pixels_to_remove": int(donations_pixels_to_remove),
            "chunk_height_reduction_factor": float(donations_chunk_height_reduction_factor),
            "export_server_data_flag": bool(donations_export_server_data_flag),
            "export_clan_data_flag" : bool(donations_export_clan_data_flag),
            "min_score_filter": Globals.MIN_SCORE_DONATIONS,
            "numbers_filtered": Globals.NUMBERS_FILTERED_2,
            "report_type": donations_report_type
            })

        df_all, df_clan = get_data()
        
        if df_clan is not None:
            df_clan = manual_confirmation_of_low_scores(df_clan)
            df_all.update(df_clan)
    
        df_all, df_clan = get_final_names(df_all, df_clan)
    
        # Save CSV
        save_with_prompt(df_all, df_clan)
        
        
    ### VS ###
    if vs_flag == True:
        
        Globals.reset()
        
        Globals.variables_state.update({
            "metric_name": str(vs_metric),
            "file_name": str(vs_file_name),
            "server": vs_server,
            "enemy_server": vs_enemy_server,
            "num_players": int(vs_num_players),
            "clan_tag_location": str(vs_clan_tag_location) if vs_clan_tag_location else None,
            "clan_tag_to_filteron": str(vs_clan_tag_to_filteron) if vs_clan_tag_to_filteron else None,
            "default_clan_tag": str(vs_clan_tag_to_filteron) if not vs_clan_tag_location else None,
            "player_order_continuous": bool(vs_player_order_continuous),
            "rank_strip_start": int(vs_rank_strip_start),
            "rank_strip_end": int(vs_rank_strip_end),
            "pixels_to_remove": int(vs_pixels_to_remove),
            "chunk_height_reduction_factor": float(vs_chunk_height_reduction_factor),
            "export_server_data_flag": bool(vs_export_server_data_flag),
            "export_clan_data_flag" : bool(vs_export_clan_data_flag),
            "min_score_filter": Globals.MIN_SCORE_VS,
            "numbers_filtered": Globals.NUMBERS_FILTERED_2,
            "report_type": vs_report_type
            })

        df_all, df_clan = get_data()
        
        if df_clan is not None:
            df_clan = manual_confirmation_of_low_scores(df_clan)
            df_all.update(df_clan)
            
        df_all, df_clan = get_final_names(df_all, df_clan)
    
        # Save CSV
        save_with_prompt(df_all, df_clan)
        
    ### serverwars ###
    if serverwars_flag == True:
        
        Globals.reset()
        
        Globals.variables_state.update({
            "metric_name": str(serverwars_metric),
            "file_name": str(serverwars_file_name),
            "server": serverwars_server,
            "enemy_server": serverwars_enemy_server,
            "num_players": int(serverwars_num_players),
            "clan_tag_location": str(serverwars_clan_tag_location) if serverwars_clan_tag_location else None,
            "clan_tag_to_filteron": str(serverwars_clan_tag_to_filteron) if serverwars_clan_tag_to_filteron else None,
            "default_clan_tag": str(serverwars_clan_tag_to_filteron) if not serverwars_clan_tag_location else None,
            "player_order_continuous": bool(serverwars_player_order_continuous),
            "rank_strip_start": int(serverwars_rank_strip_start),
            "rank_strip_end": int(serverwars_rank_strip_end),
            "pixels_to_remove": int(serverwars_pixels_to_remove),
            "chunk_height_reduction_factor": float(serverwars_chunk_height_reduction_factor),
            "export_server_data_flag": bool(serverwars_export_server_data_flag),
            "export_clan_data_flag" : bool(serverwars_export_clan_data_flag),
            "min_score_filter": Globals.MIN_SCORE_SERVERWARS,
            "numbers_filtered": Globals.NUMBERS_FILTERED_2,
            "report_type": serverwars_report_type
            })

        df_all, df_clan = get_data()
        
        if df_clan is not None:
            df_clan = manual_confirmation_of_low_scores(df_clan)
            df_all.update(df_clan)
    
        df_all, df_clan = get_final_names(df_all, df_clan)
    
        # Save CSV
        save_with_prompt(df_all, df_clan)       
    
