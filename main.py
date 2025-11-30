import gradio as gr
import mlx_whisper
import yt_dlp
import os

def download_audio(url):
    """Download audio from a URL using yt-dlp."""
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
    # 1. Determine Input Source
    target_file = None
    if audio_file is not None:
        target_file = audio_file
    elif url_input and url_input.strip() != "":
        target_file = download_audio(url_input)
    else:
        return "Please upload a file OR paste a URL."

    # 2. Handle 'Auto' Language
    lang_param = None if language == "Auto" else language

    print(f"Processing... Model: {model_id} | Lang: {lang_param} | Task: {task}")

    try:
        # 3. Run Transcription
        result = mlx_whisper.transcribe(
            target_file,
            path_or_hf_repo=model_id,
            language=lang_param,
            task=task,
            initial_prompt=initial_prompt
        )
        
        # 4. Cleanup downloads
        if url_input and target_file and "downloads/" in target_file:
            if os.path.exists(target_file):
                os.remove(target_file)
            
        return result["text"]
    except Exception as e:
        return f"Error: {str(e)}"

def build_interface():
    with gr.Blocks(title="MLX Whisper WebUI") as demo:
        gr.Markdown("# MLX Whisper (Advanced)")
        gr.Markdown("Supports Video URLs and Multiple Models.")
        
        with gr.Row():
            with gr.Column():
                # Tabs
                with gr.Tab("Upload File"):
                    audio_input = gr.Audio(type="filepath", label="Upload Audio/Video")
                with gr.Tab("Paste URL"):
                    url_input = gr.Textbox(label="Video URL", placeholder="YouTube, TikTok, X...")

                # --- CORRECTED MODEL NAMES HERE ---
                model_input = gr.Dropdown(
                    choices=[
                        "mlx-community/whisper-large-v3-turbo", 
                        "mlx-community/whisper-large-v3-mlx",   # Fixed
                        "mlx-community/whisper-medium-mlx",     # Fixed
                        "mlx-community/whisper-base-mlx"        # Fixed
                    ],
                    value="mlx-community/whisper-large-v3-turbo",
                    label="Model Selection (Try 'Medium' if Translate fails)"
                )

                # Language Controls
                language_input = gr.Dropdown(
                    choices=["Auto", "zh", "en", "ja", "ko", "es", "fr", "de"], 
                    value="zh", 
                    label="Source Language (Force 'zh' for Chinese)"
                )
                
                # Task Controls
                task_input = gr.Radio(
                    choices=["transcribe", "translate"], 
                    value="transcribe", 
                    label="Task (Translate = Always to English)"
                )
                
                prompt_input = gr.Textbox(
                    label="Initial Prompt", 
                    placeholder="Context hint (e.g., 'Translate to English' or '繁體中文')"
                )
                
                submit_btn = gr.Button("Submit", variant="primary")
            
            with gr.Column():
                output_text = gr.Textbox(label="Result", lines=15, show_copy_button=True)

        submit_btn.click(
            fn=process_audio, 
            inputs=[audio_input, url_input, model_input, language_input, task_input, prompt_input], 
            outputs=output_text
        )

    return demo

if __name__ == "__main__":
    demo = build_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)
