import argparse
import json
import os
import time


def init_gemini():
    with open("/home/qasim/Projects/graph-robotics/api_keys/gemini_key.txt") as f:
        GOOGLE_API_KEY = f.read().strip()
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(
        "/home/qasim/Projects/graph-robotics/api_keys/total-byte-432318-q3-78e6d4aa6497.json"
    )


def call_gemini(model, prompt):
    response = None
    try:
        response = model.generate_content(prompt)
    except Exception as e:
        print("API Error:", e)
        time.sleep(60)
        response = model.generate_content(prompt)
    try:
        response = response.text
    except:
        response = "No response"
    return response
