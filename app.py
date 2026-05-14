import gradio as gr
import shutil
import tempfile
import logging
from pathlib import Path
from config import Config
from logic import process_audio, cleanup_session

#Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#JavaScript for Warning 
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
    gr.Markdown("Upload audio to extract engaging moments. **Privacy Note:** Files are deleted when you click 'Done' or close the tab.")
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_in = gr.Audio(label="Upload Audio", type="filepath")
            run_btn = gr.Button("Process Audio", variant="primary")
            
        with gr.Column(scale=2):
            transcript = gr.Textbox(label="Transcript", lines=20)
    with gr.Row():
        status = gr.Textbox(label="Status", placeholder="Waiting for upload")

    gr.Markdown("Selected Viral Clips")
    with gr.Row():
        c1 = gr.Audio(label="Clip 1")
        c2 = gr.Audio(label="Clip 2")
        c3 = gr.Audio(label="Clip 3")
    with gr.Row():
        done_btn = gr.Button("Done", variant="stop")

    #Event Listeners
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
    demo.unload(fn=cleanup_session)

if __name__ == "__main__":
    #Validate Secrets
    try:
        Config.validate_env()
    except Exception as e:
        logger.error(f"Setup Error: {e}")
        import sys
        sys.exit(1)

    #Pre-launch temp cleanup
    temp_path = Path(tempfile.gettempdir())
    for old_session in temp_path.glob("session_*"):
        shutil.rmtree(old_session, ignore_errors=True)
        
    #Launch with max protection against schema bugs
    demo.queue(max_size=3).launch(show_api=False, server_name="0.0.0.0")