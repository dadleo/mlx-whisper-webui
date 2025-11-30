import gradio as gr
import mlx_whisper

# Use the same model as the original
HF_REPO = "mlx-community/whisper-turbo"

def transcribe(audio, language, task, initial_prompt):
    if audio is None:
        return None
    
    print(f"Processing... Language: {language}, Task: {task}")

    # If "Auto" is selected, pass None. Otherwise pass the code (e.g., "zh")
    target_lang = None if language == "Auto" else language

    try:
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=HF_REPO,
            language=target_lang,
            task=task,
            initial_prompt=initial_prompt
        )
        return result["text"]
    except Exception as e:
        return f"Error: {str(e)}"

# Build the Interface with the new controls
def build_interface():
    iface = gr.Interface(
        fn=transcribe,
        inputs=[
            gr.Audio(type="filepath", label="Audio"),
            
            # 1. Force Language (Fixes Chinese->Japanese issue)
            gr.Dropdown(
                choices=["Auto", "zh", "en", "ja", "ko", "es", "fr", "de"], 
                value="zh", 
                label="Language (Select 'zh' for Chinese)"
            ),
            
            # 2. Select Task
            gr.Radio(
                choices=["transcribe", "translate"], 
                value="transcribe", 
                label="Task"
            ),
            
            # 3. Optional Prompt to help the AI
            gr.Textbox(
                label="Initial Prompt", 
                value="以下是普通话的句子。",
                placeholder="Hint for the AI (e.g., This is Chinese)"
            )
        ],
        outputs="text",
        title="MLX Whisper WebUI (Fixed)",
        description="Select 'zh' in the dropdown to stop Japanese hallucinations."
    )
    return iface

if __name__ == "__main__":
    demo = build_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)