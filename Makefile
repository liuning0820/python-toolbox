# Define the Python scripts to run with Streamlit
SCRIPTS = ollama_url_summarizer.py ollama_chat_with_pdf.py youtube_video_summarizer.py

# Default target to run all Streamlit apps
all: run

# Target to run all Streamlit apps concurrently
run:
	@echo "Starting Streamlit apps..."
	@for script in $(SCRIPTS); do \
		echo "Running $$script..."; \
		streamlit run $$script & \
	done
	@echo "All Streamlit apps are running."

# Target to stop all Streamlit processes
stop:
	@echo "Stopping Streamlit apps..."
	@pkill -f "streamlit run"
	@echo "All Streamlit apps stopped."

# Clean up any temporary files (if needed)
clean:
	@echo "Cleaning up..."
	@rm -rf *.pyc __pycache__/

.PHONY: all run stop clean
