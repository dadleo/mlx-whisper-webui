import gradio as gr
import mlx_whisper
import yt_dlp
import os
import math

def format_timestamp(seconds):
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = math.floor(seconds / 3600)
    minutes = math.floor((seconds % 3600) / 60)
    secs = math.floor(seconds % 60)
    millis = math.floor((seconds - math.floor(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def generate_srt(segments):
    """Convert Whisper segments to SRT string."""
    srt_content = ""
    for i, segment in enumerate(segments):
        start = format_timestamp(segment['start'])
        end = format_timestamp(segment['end'])
        text = segment['text'].strip()
        srt_content += f"{i+1}\n{start} --> {end}\n{text}\n\n"
    return srt_content

def download_audio(url):
    print(f"Downloading from: {url}")
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'downloads/%(id)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
        'no_warnings': True,
    }
    os.makedirs("downloads", exist_ok=True)
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            final_filename = filename.rsplit('.', 1)[0] + ".mp3"
            return final_filename
    except Exception as e:
        raise gr.Error(f"Download failed: {str(e)}")

def process_audio(audio_file, url_input, model_id, language, task, initial_prompt):
    target_file = None
    
    # Check URL first, then File
    if url_input and url_input.strip() != "":
        target_file = download_audio(url_input)
    elif audio_file is not None:
        target_file = audio_file
    else:
        return "Please upload a file OR paste a URL.", None

    # Handle 'Auto' Language
    lang_param = None if language == "Auto" else language

    print(f"Processing... Model: {model_id} | Lang: {lang_param} | Task: {task}")

    try:
        # Run Transcription
        result = mlx_whisper.transcribe(
            target_file,
            path_or_hf_repo=model_id,
            language=lang_param,
            task=task,
            initial_prompt=initial_prompt
        )
        
        text_output = result["text"]
        
        # --- NEW: Generate SRT File ---
        srt_content = generate_srt(result["segments"])
        srt_filename = "subtitles.srt"
        with open(srt_filename, "w", encoding="utf-8") as f:
            f.write(srt_content)
        
        # Cleanup downloads
        if url_input and target_file and "downloads/" in target_file:
            if os.path.exists(target_file):
                os.remove(target_file)
            
        return text_output, srt_filename

    except Exception as e:
        return f"Error: {str(e)}", None

def build_interface():
    with gr.Blocks(title="MLX Whisper WebUI") as demo:
        gr.Markdown("# 🍎 MLX Whisper WebUI")
        
        with gr.Accordion("📖 Usage Guide & Tips", open=False):
            gr.Markdown("... (Tips are hidden) ...")

        with gr.Row():
            with gr.Column():
                with gr.Tab("Upload File") as tab_file:
                    audio_input = gr.Audio(type="filepath", label="Upload Audio/Video")
                with gr.Tab("Paste URL") as tab_url:
                    url_input = gr.Textbox(label="Video URL", placeholder="YouTube, TikTok, X...")

                model_input = gr.Dropdown(
                    choices=[
                        "mlx-community/whisper-large-v3-turbo", 
                        "mlx-community/whisper-large-v3-mlx",   
                        "mlx-community/whisper-medium-mlx",     
                        "mlx-community/whisper-base-mlx"        
                    ],
                    value="mlx-community/whisper-large-v3-turbo",
                    label="Model Selection"
                )

                language_input = gr.Dropdown(
                    choices=["Auto", "zh", "en", "ja", "ko", "es", "fr", "de"], 
                    value="zh", 
                    label="Source Language"
                )
                task_input = gr.Radio(["transcribe", "translate"], value="transcribe", label="Task")
                prompt_input = gr.Textbox(label="Initial Prompt", placeholder="Context hint")
                
                submit_btn = gr.Button("Submit", variant="primary")
            
            with gr.Column():
                # Output 1: Text
                output_text = gr.Textbox(label="Transcription Text", lines=15, show_copy_button=True)
                
                # Output 2: File Download (The real "SRT" feature)
                output_file = gr.File(label="Download Subtitles (.srt)")

        # Clear inputs on tab switch
        tab_file.select(fn=lambda: None, outputs=url_input)
        tab_url.select(fn=lambda: None, outputs=audio_input)

        # Logic with 2 Outputs
        submit_btn.click(
            fn=process_audio, 
            inputs=[audio_input, url_input, model_input, language_input, task_input, prompt_input], 
            outputs=[output_text, output_file] # Returning Text AND File
        )

    return demo

if __name__ == "__main__":
    demo = build_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)
