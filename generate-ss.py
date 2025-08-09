import base64
import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# SPEEDCAT_URL is expected to be set in the .env file
# 登录SPEEDCAT，复制 V2Ray 订阅链接,各软件通用，全SS协议
ss_url=os.getenv("SPEEDCAT_URL") 

# Fetch the encoded string from the URL
try:
    response = requests.get(ss_url)
    response.raise_for_status()
    encoded_string = response.text
    # print(f"Encoded string fetched: {encoded_string}")
except Exception as e:
    print(f"Error fetching encoded string: {e}")
    encoded_string = ""

try:
    decoded_bytes = base64.b64decode(encoded_string)
    decoded_string = decoded_bytes.decode("utf-8")
    print(decoded_string)
except Exception as e:
    print(f"Error decoding: {e}")