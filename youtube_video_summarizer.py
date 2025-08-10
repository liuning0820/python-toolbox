import json
import streamlit as st
import os, subprocess
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import whisper
import hashlib
from dotenv import load_dotenv
from openai import OpenAI


# Load environment variables from .env
load_dotenv()

# ---------- CONFIG ----------
BASE_CACHE_DIR = ".cache"

FPS = 0.5  # Extract 1 frame every 2 seconds

OLLAMA_LLM_MODEL=os.getenv("OLLAMA_LLM_MODEL", "qwen3:32b")
OLLAMA_LLM_BASE_URL = os.getenv("OLLAMA_LLM_BASE_URL", "http://localhost:11434") + '/v1'
# OLLAMA_LLM_MODEL='gemma3:1b'
# OLLAMA_LLM_BASE_URL='http://localhost:11434/v1'

# ---------- HELPERS ----------
@st.cache_data
def download_youtube(url, video_file):
    if not os.path.exists(video_file):
        st.write("📥 Downloading video...")
        subprocess.run(["yt-dlp", "-f", "mp4", "-o", video_file, url])
    else:
        st.write("🎥 Video already downloaded")


@st.cache_resource
def load_whisper_model():
    return whisper.load_model("small")

@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    return processor, model


def get_video_hash(url):
    return hashlib.sha256(url.encode()).hexdigest()[:12]

def transcribe_video(video_file, transcript_file):
    if os.path.exists(transcript_file):
        return load_transcript(transcript_file)

    model = load_whisper_model()
    result = model.transcribe(video_file,verbose=True)
    with open(transcript_file, "w") as f:
        f.write(result["text"])
    return result["text"]

def load_transcript(transcript_file):
    try:
        with open(transcript_file, "r") as f:
            text = f.read()
        return text
    except FileNotFoundError:
        return None

def extract_frames(frame_dir):
    if not os.path.exists(frame_dir) or len(os.listdir(frame_dir)) == 0:
        st.write("🖼️ Extracting frames...")
        os.makedirs(frame_dir, exist_ok=True)
        subprocess.run(["ffmpeg", "-i", VIDEO_FILE, "-vf", f"fps={FPS}", f"{frame_dir}/frame_%04d.jpg", "-hide_banner", "-loglevel", "error"])

def caption_frames(frame_dir, captions_file):
    if os.path.exists(captions_file):
        return json.load(open(captions_file, "r"))

    st.write("🧠 Generating captions with BLIP...")
    processor, model = load_blip_model()
    captions = []
    for fname in sorted(os.listdir(frame_dir)):
        if fname.endswith(".jpg"):
            image = Image.open(os.path.join(frame_dir, fname)).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=30)
            caption = processor.decode(out[0], skip_special_tokens=True)
            captions.append((fname, caption))
    with open(captions_file, "w") as f:
        json.dump(captions, f)
    return captions

def summarize_with_ollama(captions, transcript):
    st.write("💬 Summarizing with Ollama...")
    caption_text = "\n".join([c for _, c in captions])
    prompt = f"""You are an expert at summarizing videos. Please summarize the key points from this video. Here are captions from sampled frames:

{caption_text}

Here is the transcript of the video:

{transcript}

"""
    client = OpenAI(api_key='ollama', base_url=OLLAMA_LLM_BASE_URL)  # Adjust base_url if needed
    response = client.chat.completions.create(
        model=OLLAMA_LLM_MODEL,  # You can change to your preferred model
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# ---------- UI ----------
st.set_page_config(page_title="Video Summarizer", layout="centered")
st.title("🎬 YouTube Video Summarizer (Local, Private)")
url = st.text_input("Paste YouTube Video URL:")

device = "mps" if torch.backends.mps.is_available() else "cpu"

if st.button("Start"):
    try:
        video_hash = get_video_hash(url)
        cache_dir = os.path.join(BASE_CACHE_DIR, video_hash)
        os.makedirs(cache_dir, exist_ok=True)
        VIDEO_FILE = os.path.join(cache_dir, "video.mp4")
        TRANSCRIPT_FILE = os.path.join(cache_dir, "transcript.txt")
        FRAME_DIR = os.path.join(cache_dir, "frames")
        CAPTIONS_FILE = os.path.join(cache_dir, "captions.json")

        # Step 1: Download & extract
        download_youtube(url, VIDEO_FILE)
        extract_frames(FRAME_DIR)

        # Step 2: Caption
        captions = caption_frames(FRAME_DIR, CAPTIONS_FILE)
        st.subheader("📝 Captions from Frames")
        with st.expander("View Frame Images", expanded=False):
            for fname, caption in captions:
                st.image(os.path.join(FRAME_DIR, fname), width=640)
                st.caption(caption)

        transcript = transcribe_video(VIDEO_FILE, TRANSCRIPT_FILE)
        st.subheader("📝 Transcript")
        # st.markdown(transcript)
        st.text_area("Transcript", transcript, height=300)

        # Step 3: Summarize
        summary = summarize_with_ollama(captions, transcript)
        st.subheader("📚 Summary")
        st.markdown(summary)

    except Exception as e:
        st.error(f"Something went wrong: {e}")


# The core functionality of the script, which is to download a YouTube video, 
# extract frames, generate captions for those frames using the BLIP model, 
# and then summarize the content using Ollama.