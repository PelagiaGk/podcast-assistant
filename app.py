import gradio as gr
import shutil
import tempfile
import logging
from pathlib import Path
from config import Config
from logic import process_media, cleanup_session 

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# JavaScript for Warning 
warning_js = """
function() {
    window.onbeforeunload = function() {
        return "Warning: Your session data and clips will be permanently deleted if you leave this page.";
    };
}
"""

def reset_ui():
    """Returns empty values to reset the UI components after cleanup."""
    # Matches the 7 elements expected to perfectly clear inputs, outputs, and state
    return "", "Session Cleaned", None, None, None, None, ""

# UI Interface
# NOTE: Adjusted delete_cache to prevent sweeping files mid-processing for large files
with gr.Blocks(theme=gr.themes.Soft(), js=warning_js, delete_cache=(3600, 3600)) as demo:
    session_state = gr.State("")
    
    gr.Markdown("# Podcast Assistant - Viral Clips Extractor")
    gr.Markdown("Upload **audio** or **video** to extract engaging moments. **Privacy Note:** Files are deleted when you click 'Done' or close the tab.")
    
    with gr.Row():
        with gr.Column(scale=1):
            media_in = gr.File(
                label="Upload Audio or Video", 
                file_types=["audio", "video"],
                type="filepath"
            )
            run_btn = gr.Button("Process Media", variant="primary")
            
        with gr.Column(scale=2):
            transcript = gr.Textbox(label="Transcript", lines=12)

    with gr.Row():
        status = gr.Textbox(label="Status", placeholder="Waiting for upload")

    gr.Markdown("### Selected Viral Clips")
    with gr.Row():
        c1 = gr.File(label="Clip 1")
        c2 = gr.File(label="Clip 2")
        c3 = gr.File(label="Clip 3")

    with gr.Row():
        done_btn = gr.Button("Done", variant="stop")

    # --- THE CRITICAL FIX ---
    # Added 'session_state' twice at the end to map BOTH the final unpacked clip, 
    # AND the session directory string seamlessly.
    run_btn.click(
        fn=process_media, 
        inputs=media_in, 
        outputs=[transcript, status, c1, c2, c3, session_state, session_state]
    )
    
    # Cleanup logic: Deletes files then clears the UI
    done_btn.click(
        fn=cleanup_session,
        inputs=session_state
    ).then(
        fn=reset_ui,
        outputs=[transcript, status, c1, c2, c3, media_in, session_state]
    )

    demo.unload(fn=cleanup_session)

if __name__ == "__main__":
    # Validate Secrets
    try:
        Config.validate_env()
    except Exception as e:
        logger.error(f"Setup Error: {e}")
        import sys
        sys.exit(1)

    # Pre-launch temp cleanup for safety
    temp_path = Path(tempfile.gettempdir())
    for old_session in temp_path.glob("session_*"):
        shutil.rmtree(old_session, ignore_errors=True)
        
    # Launch
    demo.queue(max_size=3).launch(show_api=False, server_name="0.0.0.0")