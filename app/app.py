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
    
    # 2. Track the last RL action if it was a JITAI
    last_arm = -1
    if "[JITAI" in mode_response:
        # Determine which arm was selected (Reverse lookup for feedback)
        for i, arm_name in enumerate(harness.rl_policy.arms):
            if arm_name.split()[0] in mode_response: # e.g., "Breathing" in "[JITAI - Breathing]"
                last_arm = i
                break

    # 3. Simulate RAG context injection using RCR
    if "CLINICAL_RAG" in mode_response or "GENERAL_SUPPORT" in mode_response:
        context = retriever.retrieve(user_input, vitals_z_scores)
        final_output = f"[{mode_response}]\n[RCR Context Loaded: {context}]\n\nAgent: I understand what you are going through. Based on your current vitals and clinical guidelines, I suggest we explore this together..."
    else:
        final_output = mode_response
        
    chat_history.append((user_input if user_input else "[SILENT Z-SCORE SYNC]", final_output))
    return chat_history, chat_history, last_arm

def update_rl(last_arm, feedback_type, chat_history):
    if last_arm != -1:
        reward = 1.0 if feedback_type == "Helpful 👍" else 0.0
        harness.rl_policy.update(last_arm, reward)
        arm_name = harness.rl_policy.arms[last_arm]
        status = f"RL Updated: {arm_name} received reward {reward}. Current Alphas: {harness.rl_policy.alphas.round(1)}"
        chat_history.append((None, f"🧬 [System Note] {status}"))
    return chat_history, chat_history, -1

# Gradio Interface (Bio-Digital Twin Dashboard)
with gr.Blocks(title="HybridSense Bio-Digital Twin") as demo:
    gr.Markdown("# 🧠 HybridSense-Agentic: Bio-Digital Twin Prototype")
    gr.Markdown("Interact with the Agent. Adjust the **Z-score deviation sliders** to simulate how the system proactively responds.")
    
    last_rl_arm = gr.State(-1)
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Therapeutic Interaction")
            with gr.Row():
                user_text = gr.Textbox(label="User Input", placeholder="E.g., I'm having trouble breathing...", scale=4)
                submit_btn = gr.Button("Send / Sync Vitals", variant="primary", scale=1)
            
            with gr.Row():
                gr.Markdown("#### JITAI Feedback (RL Reinforcement):")
                feedback_pos = gr.Button("Helpful 👍", size="sm")
                feedback_neg = gr.Button("Not Helpful 👎", size="sm")
            
        with gr.Column(scale=1):
            gr.Markdown("### 🫀 Physiological Deviation (Z-Scores)")
            slider_hr = gr.Slider(minimum=-1.0, maximum=4.0, value=0.0, step=0.1, label="HR Z-Score [>3.0 = Crisis]")
            slider_hrv = gr.Slider(minimum=-4.0, maximum=1.0, value=0.0, step=0.1, label="HRV Z-Score [<-2.0 = High Stress JITAI]")
            slider_wakeups = gr.Slider(minimum=-1.0, maximum=4.0, value=0.0, step=0.1, label="Wakeup Z-Score [>2.0 = Sleep Disruption]")
            
    history = gr.State([])
    
    submit_btn.click(
        fn=process_interaction,
        inputs=[user_text, slider_hr, slider_hrv, slider_wakeups, history],
        outputs=[chatbot, history, last_rl_arm]
    )

    feedback_pos.click(
        fn=update_rl,
        inputs=[last_rl_arm, gr.State("Helpful 👍"), history],
        outputs=[chatbot, history, last_rl_arm]
    )
    
    feedback_neg.click(
        fn=update_rl,
        inputs=[last_rl_arm, gr.State("Not Helpful 👎"), history],
        outputs=[chatbot, history, last_rl_arm]
    )


if __name__ == "__main__":
    print("Launching Bio-Digital Twin Gradio App...")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
