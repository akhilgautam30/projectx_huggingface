import gradio as gr
from model_utils import predict_personality

# Create Gradio interface
iface = gr.Interface(
    fn=predict_personality,
    inputs=gr.Textbox(lines=5, label="Enter text for personality prediction"),
    outputs=gr.Label(num_top_classes=5, label="Personality Traits"),
    title="Personality Prediction with RoBERTa",
    description="Enter some text to predict personality traits using a fine-tuned RoBERTa model."
)

# Launch the interface
if __name__ == "__main__":
    iface.launch()
