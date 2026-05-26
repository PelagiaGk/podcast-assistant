import gradio as gr
import shutil
import tempfile
import logging
from pathlib import Path
from config import Config
from logic import process_media, cleanup_session 

#Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#JavaScript for Warning 
warning_js = """
function() {
    window.onbeforeunload = function() {
        return "Warning: Your session data and clips will be permanently deleted if you leave this page.";
    };
}
"""

#HTML Loading Spinner and Animations
SPINNER_HTML = """
<div style="display: flex; align-items: center; gap: 12px; margin: 15px 0; padding: 10px; background-color: rgba(138, 92, 246, 0.1); border-radius: 8px;">
    <div style="
        width: 24px; 
        height: 24px; 
        border: 3px solid #8a5cf6; 
        border-top-color: transparent; 
        border-radius: 50%; 
        animation: spin 1s linear infinite;
    "></div>
    <span style="color: #8a5cf6; font-weight: 600; font-size: 1.1em;">Analyzing themes & extracting viral segments...</span>
</div>
<style>
@keyframes spin {
    to { transform: rotate(360deg); }
}
</style>
"""

def show_loading():
    """Instantly reveals the spinner loop and clears out old asset boxes."""
    return (
        gr.HTML(value=SPINNER_HTML, visible=True),
        gr.File(visible=False),
        gr.File(visible=False),
        gr.File(visible=False)
    )

def handle_pipeline_results(transcript_text, path1, path2, path3, sess1, sess2):
    """Evaluates pipeline outputs and sets visibility to match the extracted files."""
    return (
        gr.HTML(visible=False), 
        transcript_text,
        gr.File(value=path1, visible=True if path1 else False),
        gr.File(value=path2, visible=True if path2 else False),
        gr.File(value=path3, visible=True if path3 else False),
        sess1,
        sess2
    )

def reset_ui():
    """Returns empty values to reset the UI components after cleanup."""
    return (
        "",                       
        gr.File(value=None, visible=False),  
        gr.File(value=None, visible=False),  
        gr.File(value=None, visible=False),  
        None,                    
        "",                       
        gr.HTML(visible=False)    
    )

#UI Interface
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

    gr.Markdown("### Selected Viral Clips")
    
    #Hidden components 
    spinner = gr.HTML(value=SPINNER_HTML, visible=False)
    
    with gr.Row():
        c1 = gr.File(label="Clip 1", visible=False)
        c2 = gr.File(label="Clip 2", visible=False)
        c3 = gr.File(label="Clip 3", visible=False)

    with gr.Row():
        done_btn = gr.Button("Done", variant="stop")

    run_btn.click(
        fn=show_loading,
        inputs=None,
        outputs=[spinner, c1, c2, c3]
    ).then(
        fn=process_media, 
        inputs=media_in, 
        outputs=[transcript, c1, c2, c3, session_state, session_state]
    ).then(
        fn=handle_pipeline_results,
        inputs=[transcript, c1, c2, c3, session_state, session_state],
        outputs=[spinner, transcript, c1, c2, c3, session_state, session_state]
    )
    
    #Deletes files, clears the UI 
    done_btn.click(
        fn=cleanup_session,
        inputs=session_state
    ).then(
        fn=reset_ui,
        outputs=[transcript, c1, c2, c3, media_in, session_state, spinner]
    )

    demo.unload(fn=cleanup_session)

if __name__ == "__main__":
    try:
        Config.validate_env()
    except Exception as e:
        logger.error(f"Setup Error: {e}")
        import sys
        sys.exit(1)

    #Pre-launch cleanup
    temp_path = Path(tempfile.gettempdir())
    for old_session in temp_path.glob("session_*"):
        shutil.rmtree(old_session, ignore_errors=True)
        
    #Launch
    demo.queue(max_size=3).launch(server_name="0.0.0.0")