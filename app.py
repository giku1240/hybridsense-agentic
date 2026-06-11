import gradio as gr
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.safety.harness import AgentHarness
from src.rag.vector_store import PhysioFAISSRetriever

# Initialize core system components
harness = AgentHarness()
retriever = PhysioFAISSRetriever()

def process_interaction(user_input, hr_z, hrv_z, wakeup_z, chat_history):
    vitals_z_scores = {
        'hr_z_score': hr_z,
        'hrv_z_score': hrv_z,
        'wakeup_z_score': wakeup_z
    }
    
    # 1. Route through Agentic Harness (Safety & JITAI Check using Z-scores)
    mode_response = harness.process_query(user_input, vitals_z_scores)
    
    # 2. Simulate RAG context injection using RCR
    if "CLINICAL_RAG" in mode_response or "GENERAL_SUPPORT" in mode_response:
        context = retriever.retrieve(user_input, vitals_z_scores)
        final_output = f"[{mode_response}]\n[RCR Context Loaded: {context}]\n\nAgent: I understand what you are going through. Based on your current vitals and clinical guidelines, I suggest we explore this together..."
    else:
        # Crisis or Proactive JITAI handled directly by the harness
        final_output = mode_response
        
    chat_history.append((user_input if user_input else "[SILENT Z-SCORE SYNC]", final_output))
    return chat_history, chat_history

# Gradio Interface (Bio-Digital Twin Dashboard)
with gr.Blocks(title="HybridSense Bio-Digital Twin") as demo:
    gr.Markdown("# 🧠 HybridSense-Agentic: Bio-Digital Twin Prototype")
    gr.Markdown("Interact with the Agent. Adjust the **Z-score deviation sliders** to simulate how the system proactively responds when your physiological state deviates from your personal baseline.")
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Therapeutic Interaction")
            user_text = gr.Textbox(label="User Input (Leave blank to simulate silent physiological monitoring)", placeholder="E.g., I'm having trouble breathing...")
            submit_btn = gr.Button("Send / Sync Vitals")
            
        with gr.Column(scale=1):
            gr.Markdown("### 🫀 Physiological Deviation (Z-Scores)")
            slider_hr = gr.Slider(minimum=-1.0, maximum=4.0, value=0.0, step=0.1, label="HR Z-Score [>3.0 = Crisis]")
            slider_hrv = gr.Slider(minimum=-4.0, maximum=1.0, value=0.0, step=0.1, label="HRV Z-Score [<-2.0 = High Stress JITAI]")
            slider_wakeups = gr.Slider(minimum=-1.0, maximum=4.0, value=0.0, step=0.1, label="Wakeup Z-Score [>2.0 = Sleep Disruption]")
            
    history = gr.State([])
    
    submit_btn.click(
        fn=process_interaction,
        inputs=[user_text, slider_hr, slider_hrv, slider_wakeups, history],
        outputs=[chatbot, history]
    )

if __name__ == "__main__":
    print("Launching Bio-Digital Twin Gradio App...")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
