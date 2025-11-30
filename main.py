import gradio as gr
import mlx_whisper
import yt_dlp
import os

# Define the model
HF_REPO = "mlx-community/whisper-turbo"

def download_audio(url):
    """Download audio from a URL using yt-dlp."""
    print(f"Downloading from: {url}")
    
    # Configure yt-dlp to download best audio and convert to mp3
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'downloads/%(id)s.%(ext)s',  # Save to a subfolder
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
        'no_warnings': True,
    }
    
    # Create downloads directory if not exists
    os.makedirs("downloads", exist_ok=True)
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            # yt-dlp changes extension after conversion, assume mp3
            final_filename = filename.rsplit('.', 1)[0] + ".mp3"
            return final_filename
    except Exception as e:
        raise gr.Error(f"Download failed: {str(e)}")

def process_audio(audio_file, url_input, language, task, initial_prompt):
    # Determine source: File or URL?
    target_file = None
    
    # Logic: Prioritize File if both are present, or use URL
    if audio_file is not None:
        target_file = audio_file
    elif url_input and url_input.strip() != "":
        target_file = download_audio(url_input)
    else:
        return "Please upload a file OR paste a URL."

    print(f"Transcribing... Lang: {language} | Task: {task}")

    # Handle "Auto" language
    lang_param = None if language == "Auto" else language

    try:
        result = mlx_whisper.transcribe(
            target_file,
            path_or_hf_repo=HF_REPO,
            language=lang_param,
            task=task,
            initial_prompt=initial_prompt
        )
        
        # Optional: Cleanup the downloaded file to save space
        if url_input and target_file and "downloads/" in target_file:
            if os.path.exists(target_file):
                os.remove(target_file)
            
        return result["text"]
    except Exception as e:
        return f"Error during transcription: {str(e)}"

def build_interface():
    with gr.Blocks(title="MLX Whisper WebUI (Video/Audio)") as demo:
        gr.Markdown("# MLX Whisper (File & URL Support)")
        gr.Markdown("Supports Uploads + YouTube, TikTok, X, SoundCloud, etc.")
        
        with gr.Row():
            with gr.Column():
                # Tabs for Input Source
                with gr.Tab("Upload File"):
                    audio_input = gr.Audio(type="filepath", label="Upload Audio/Video File")
                
                with gr.Tab("Paste URL"):
                    url_input = gr.Textbox(
                        label="Video URL", 
                        placeholder="https://www.youtube.com/watch?v=..."
                    )

                # Settings
                language_input = gr.Dropdown(
                    choices=["Auto", "zh", "en", "ja", "ko", "es", "fr", "de"], 
                    value="zh", 
                    label="Language (Force 'zh' for Chinese)"
                )
                task_input = gr.Radio(
                    choices=["transcribe", "translate"], 
                    value="transcribe", 
                    label="Task"
                )
                prompt_input = gr.Textbox(
                    label="Initial Prompt", 
                    value="以下是普通话的句子。",
                    placeholder="Context hint"
                )
                
                submit_btn = gr.Button("Submit", variant="primary")
            
            with gr.Column():
                output_text = gr.Textbox(label="Result", lines=12, show_copy_button=True)

        # Logic
        submit_btn.click(
            fn=process_audio, 
            inputs=[audio_input, url_input, language_input, task_input, prompt_input], 
            outputs=output_text
        )

    return demo

if __name__ == "__main__":
    demo = build_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)