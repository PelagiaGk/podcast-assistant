import gradio as gr
import shutil
import tempfile
from pathlib import Path
from config import Config
from logic import process_audio, cleanup_session

#JavaScript 
warning_js = """
function() {
    window.onbeforeunload = function() {
        return "Warning: Your session data will be deleted if you leave.";
    };
}
"""

with gr.Blocks(theme=gr.Theme.from_hub("theme-repo/STONE_Theme")) as demo:
    session_state = gr.State("")
    
    gr.Markdown("Podcast Assistant - Viral Clips Extractor")
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_in = gr.Audio(label="Upload Audio", type="filepath")
            run_btn = gr.Button("Process Audio", variant="primary")
            status = gr.Textbox(label="Status", interactive=False)
            done_btn = gr.Button("Done", variant="stop")
            
        with gr.Column(scale=2):
            transcript = gr.Textbox(label="Transcript", lines=12, interactive=False)

    gr.Markdown("Selected Viral Hooks")
    with gr.Row():
        c1 = gr.Audio(label="Hook 1", interactive=False)
        c2 = gr.Audio(label="Hook 2", interactive=False)
        c3 = gr.Audio(label="Hook 3", interactive=False)

    run_btn.click(
        process_audio, 
        inputs=audio_in, 
        outputs=[transcript, status, c1, c2, c3, session_state]
    )
    
    done_btn.click(
        cleanup_session,
        inputs=session_state,
        outputs=[transcript, status, c1, c2, c3, audio_in, session_state]
    )

    demo.unload(cleanup_session)

if __name__ == "__main__":
    Config.validate_env()
    
    #Pre-launch cleanup
    temp_path = Path(tempfile.gettempdir())
    for old_session in temp_path.glob("session_*"):
        shutil.rmtree(old_session, ignore_errors=True)
        
    demo.queue(max_size=3).launch()
#JavaScript for Browser Warning 
warning_js = """
function() {
    window.onbeforeunload = function() {
        return "Warning: Your session data and audio clips will be permanently deleted if you leave this page.";
    };
}
"""
#UI Interface 

with gr.Blocks(theme=gr.themes.Soft(), js=warning_js, delete_cache=(60, 60)) as demo:
    session_state = gr.State("")
    
    gr.Markdown("Podcast Assistant - Viral Clips Extractor")
    gr.Markdown("Upload audio to extract the most engaging moments. **Privacy Note:** Your files are processed in a temporary session and deleted when you click 'Done' or close the tab.")
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_in = gr.Audio(label="Upload Audio", type="filepath")
            run_btn = gr.Button("Process Audio", variant="primary")
            status = gr.Textbox(label="Status", interactive=False)
            done_btn = gr.Button("Done", variant="stop")
            
        with gr.Column(scale=2):
            transcript = gr.Textbox(label="Transcript", lines=12, interactive=False)

    gr.Markdown("Selected Viral Hooks")
    with gr.Row():
        c1 = gr.Audio(label="Hook 1", interactive=False)
        c2 = gr.Audio(label="Hook 2", interactive=False)
        c3 = gr.Audio(label="Hook 3", interactive=False)

    run_btn.click(
        process_audio, 
        inputs=audio_in, 
        outputs=[transcript, status, c1, c2, c3, session_state]
    )
    
    done_btn.click(
        cleanup_session,
        inputs=session_state,
        outputs=[transcript, status, c1, c2, c3, audio_in, session_state]
    )

    demo.unload(cleanup_session, inputs=session_state)

if __name__ == "__main__":
    #Validate Secret Token
    try:
        Config.validate_env()
    except EnvironmentError as e:
        logger.error(f"Setup Error: {e}")
        # In a headless environment stop execution if the token is missing
        import sys
        sys.exit(1)

    #Pre-launch cleanup
    temp_path = Path(tempfile.gettempdir())
    for old_session in temp_path.glob("session_*"):
        shutil.rmtree(old_session, ignore_errors=True)
        
    #Launch
    demo.queue(max_size=3).launch()