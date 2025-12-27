import json
import streamlit as st
import os, subprocess
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_ollama import ChatOllama
import torch
import whisper
import hashlib
from dotenv import load_dotenv

# from openai import OpenAI
from langfuse.openai import OpenAI


# Load environment variables from .env
load_dotenv()

# ---------- CONFIG ----------
BASE_CACHE_DIR = ".cache"

FPS = 0.5  # Extract 1 frame every 2 seconds

OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen3:32b")
OLLAMA_LLM_BASE_URL = os.getenv("OLLAMA_LLM_BASE_URL", "http://localhost:11434")
# OLLAMA_LLM_MODEL='gemma3:1b'
# OLLAMA_LLM_BASE_URL='http://localhost:11434'

# ---------- HELPERS ----------
@st.cache_data
def download_youtube(url, video_file):
    if not os.path.exists(video_file):
        st.write("📥 Downloading video...")
        subprocess.run(["yt-dlp", "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]", "-o", video_file, url])
    else:
        st.write("🎥 Video already downloaded")


@st.cache_resource
def load_whisper_model():
    return whisper.load_model("small")


@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=False)
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)
    return processor, model


def get_video_hash(url):
    return hashlib.sha256(url.encode()).hexdigest()[:12]


def transcribe_video(video_file, transcript_file):
    if os.path.exists(transcript_file):
        return load_transcript(transcript_file)

    model = load_whisper_model()
    result = model.transcribe(video_file, verbose=True)
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
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                VIDEO_FILE,
                "-vf",
                f"fps={FPS}",
                f"{frame_dir}/frame_%04d.jpg",
                "-hide_banner",
                "-loglevel",
                "error",
            ]
        )


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
    client = OpenAI(
        api_key="ollama", base_url=OLLAMA_LLM_BASE_URL + "/v1"
    )  # Adjust base_url if needed
    response = client.chat.completions.create(
        model=OLLAMA_LLM_MODEL,  # You can change to your preferred model
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


# ---------- SESSION STATE ----------
def init_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "suggestion_id" not in st.session_state:
        st.session_state.suggestion_id = 0
    if "video_summarized" not in st.session_state:
        st.session_state.video_summarized = False


def generate_follow_up():
    # Generate follow-up questions
    formatted_history = "\n".join(
        [
            f"{msg['role'].upper()}: {msg['message']}"
            for msg in st.session_state.chat_history
            if "message" in msg
        ]
    )

    task_prompt = f"""### Task:
Suggest 3-5 relevant follow-up questions or prompts that the user might naturally ask next in this conversation as a **user**, based on the chat history, to help continue or deepen the discussion.
### Guidelines:
- Write all follow-up questions from the user’s point of view, directed to the assistant.
- Make questions concise, clear, and directly related to the discussed topic(s).
- Only suggest follow-ups that make sense given the chat content and do not repeat what was already covered.
- If the conversation is very short or not specific, suggest more general (but relevant) follow-ups the user might ask.
- Use the conversation's primary language; default to English if multilingual.
- Your entire response must be ONLY the JSON object in the exact format specified below.
- Do not include any additional text, explanations, reasoning, or formatting before or after the JSON.
- Do not use any tags like <think>, <reason>, or similar.
- The response must start with '{{' and end with '}}'.
### Output Format:
{{ "follow_ups": ["Question 1?", "Question 2?", "Question 3?"] }}
### Chat History:
<chat_history>
{formatted_history}
</chat_history>"""

    print(task_prompt)

    follow_up_llm = OpenAI(api_key="ollama", base_url=OLLAMA_LLM_BASE_URL + "/v1")

    response = follow_up_llm.chat.completions.create(
        model=OLLAMA_LLM_MODEL, messages=[{"role": "user", "content": task_prompt}]
    )

    follow_up_response = response.choices[0].message.content

    try:
        follow_ups = json.loads(follow_up_response)["follow_ups"]
        st.session_state.suggestion_id += 1
        sugg_id = st.session_state.suggestion_id
        with st.chat_message("assistant"):
            st.markdown("Here are some suggested follow-up questions:")
            for i, q in enumerate(follow_ups):
                if st.button(q, key=f"suggest_{sugg_id}_{i}"):
                    st.session_state.fill_chat_input = q
                    st.rerun()
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "type": "suggestions",
                "id": sugg_id,
                "follow_ups": follow_ups,
            }
        )
    except Exception as e:
        follow_up_message = f"Could not generate follow-up questions. Error: {str(e)}\nRaw response: {follow_up_response}"
        with st.chat_message("assistant"):
            st.markdown(follow_up_message)
        st.session_state.chat_history.append(
            {"role": "assistant", "message": follow_up_message}
        )


# ---------- UI ----------
st.set_page_config(page_title="Video Summarizer", layout="centered")
st.title("🎬 YouTube Video Summarizer (Local, Private)")
init_session_state()
url = st.text_input("Paste YouTube Video URL:")

device = "mps" if torch.backends.mps.is_available() else "cpu"

if st.button("Start") and not st.session_state.get("video_summarized", False):
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
        st.text_area("Transcript", transcript, height=300)

        # Step 3: Summarize
        summary = summarize_with_ollama(captions, transcript)
        st.subheader("📚 Summary")
        # st.markdown(summary)  # REMOVE THIS LINE to avoid duplicate output
        # Add summary to chat history
        st.session_state.chat_history.append({"role": "assistant", "message": summary})
        if st.session_state.get("video_summarized", False):
            generate_follow_up()
        st.session_state.video_summarized = True

    except Exception as e:
        st.error(f"Something went wrong: {e}")


# Display chat history and follow-up suggestions
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        if message.get("type") == "suggestions":
            st.markdown("Here are some suggested follow-up questions:")
            for i, q in enumerate(message["follow_ups"]):
                if st.button(q, key=f"suggest_{message['id']}_{i}"):
                    st.session_state.fill_chat_input = q
                    st.rerun()
        else:
            st.markdown(message["message"])

# Persistent chat input area after summary and suggestions
if st.session_state.get("video_summarized", False):
    user_input = st.chat_input("Ask a question about the video:", key="user_chat_input")
    if user_input and (
        not st.session_state.chat_history
        or st.session_state.chat_history[-1].get("role") != "user"
        or st.session_state.chat_history[-1].get("message") != user_input
    ):
        st.session_state.chat_history.append({"role": "user", "message": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        # Use OpenAI for answering user questions
        answer_llm = OpenAI(api_key="ollama", base_url=OLLAMA_LLM_BASE_URL + "/v1")
        response = answer_llm.chat.completions.create(
            model=OLLAMA_LLM_MODEL, messages=[{"role": "user", "content": user_input}]
        )
        answer = response.choices[0].message.content
        st.session_state.chat_history.append({"role": "assistant", "message": answer})
        generate_follow_up()

if "fill_chat_input" in st.session_state and st.session_state.fill_chat_input:
    value = st.session_state.fill_chat_input
    js_code = f"""
    <script>
    function setNativeValue(element, value) {{
        let lastValue = element.value;
        element.value = value;
        let event = new Event("input", {{ target: element, bubbles: true }});
        // React 15
        event.simulated = true;
        // React 16-17
        let tracker = element._valueTracker;
        if (tracker) {{
            tracker.setValue(lastValue);
        }}
        element.dispatchEvent(event);
    }}
    var interval = setInterval(function() {{
        var chatInput = window.parent.document.querySelector('[data-testid="stChatInput"] textarea');
        if (chatInput) {{
            setNativeValue(chatInput, {json.dumps(value)});
            chatInput.focus();
            chatInput.dispatchEvent(new Event('change', {{ bubbles: true }}));
            clearInterval(interval);
        }}
    }}, 50);
    </script>
    """
    st.components.v1.html(js_code, height=0)
    del st.session_state.fill_chat_input
