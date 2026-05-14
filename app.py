import gradio as gr
import shutil
import tempfile
import logging
from pathlib import Path
from config import Config
from logic import process_audio, cleanup_session

logger = logging.getLogger(__name__)

# JavaScript for Browser Warning 
warning_js = """
function() {
    window.onbeforeunload = function() {
        return "Warning: Your session data and audio clips will be permanently deleted if you leave this page.";
    };
}
"""

with gr.Blocks(theme=gr.themes.Soft(), js=warning_js, delete_cache=(60, 60)) as demo:
    session_state = gr.State("")
    
    gr.Markdown("# Podcast Assistant - Viral Clips Extractor")
    gr.Markdown("Upload audio to extract engaging moments. **Note:** Files are deleted when you click 'Done' or close the tab.")
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_in = gr.Audio(label="Upload Audio", type="filepath")
            run_btn = gr.Button("Process Audio", variant="primary")
            status = gr.Textbox(label="Status", placeholder="Waiting for upload")
            done_btn = gr.Button("Done", variant="stop")
            
        with gr.Column(scale=2):
            transcript = gr.Textbox(label="Transcript", lines=12)

    gr.Markdown("### Selected Viral Hooks")
    with gr.Row():
        c1 = gr.Audio(label="Clip 1")
        c2 = gr.Audio(label="Clip 2")
        c3 = gr.Audio(label="Clip 3")

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

    # Corrected: unload cannot take inputs in this Gradio version
    demo.unload(cleanup_session)

if __name__ == "__main__":
    try:
        Config.validate_env()
    except Exception as e:
        logger.error(f"Setup Error: {e}")
        import sys
        sys.exit(1)

    # Pre-launch cleanup
    temp_path = Path(tempfile.gettempdir())
    for old_session in temp_path.glob("session_*"):
        shutil.rmtree(old_session, ignore_errors=True)
        
    # Launch with API docs disabled to bypass the schema bug
    demo.queue(max_size=3).launch(show_api=False)
