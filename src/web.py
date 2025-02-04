import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
from sqlalchemy.orm import sessionmaker
from src.db import engine, TextEntry, AnalysisResult, db_base
from typing import Dict, Tuple
from src.factory import ModelFactory

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db_base.metadata.create_all(bind=engine)
text = "I am feeling happy and excited about this project. I am also learning a lot from it."

def analyze_text(text: str) -> Tuple[Dict, Dict]:
    """
    Analyze input text and return emotion and education scores
    """
    model_factory = ModelFactory()
    scores = model_factory.get_model_scores(input_text=text)
    emotion_scores_dict = {scores["emotion_scores"][0][res]["label"]: scores["emotion_scores"][0][res]["score"] for res in range(4)}
    education_scores_dict = {"educational": res["score"] for res in scores["education_scores"][0]}
    return emotion_scores_dict, education_scores_dict

def fetch_data():
    """
    Fetch all entries from database
    """
    db = SessionLocal()
    try:
        all_entries = db.query(TextEntry).all()
        processed_data = []
        for row in all_entries:
            if row.results:
                result: AnalysisResult = row.results[0]
                emotion_scores = result.emotion_scores
                education_scores = result.education_scores
                scores = {**{e["label"]: e["score"] for e in emotion_scores[0]},
                         **{e["label"]: e["score"] for e in education_scores[0]}}
                processed_data.append((row.id, row.input_text, row.timestamp, scores))
        return processed_data
    finally:
        db.close()

def plot_spider_chart(scores: Dict) -> plt.Figure:
    """
    Create a clean and simple spider chart with labels offset and centered like the example.
    """
    labels = list(scores.keys())
    values = list(scores.values())

    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Close the plot
    values += values[:1]
    angles += angles[:1]

    # Create the figure and set the background color
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("#27272B")  # Set the background color of the figure
    ax.set_facecolor("#27272B")  # Set the background color of the polar plot

    # Fill area with a color that contrasts well with the background
    ax.fill(angles, values, color='#3B82F6', alpha=0.3)  # Light blue fill
    ax.plot(angles, values, color='white', linewidth=2)  # White line for the chart

    # Set the range of the radial grid and labels
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=9, color="white")  # White radial grid labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8, fontweight="medium", color="white")  # White axis labels
    ax.tick_params(axis='x', pad=13)  # Increase pad value to move labels further out


    # Add score labels at end points if score > 0, offset outward
    for i, (angle, value) in enumerate(zip(angles[:-1], values[:-1])):
        if value > 0:
            offset_factor = 1.2  # Adjust this to position labels further outward
            ax.text(
                angle, value * offset_factor, f"{value:.2f}",
                fontsize=8, fontweight="bold", color="white",  # White label text
                ha="center", va="center"
            )

    # Hide radial gridlines to match the style in the example
    ax.spines['polar'].set_visible(False)
    ax.grid(color="white", linestyle="--", linewidth=0.5)  # White dashed gridlines

    return fig


def process_text(text: str):
    """
    Process input text and return visualization and updated table
    """
    # Get scores from models
    emotion_scores, education_scores = analyze_text(text)

    # Create visualization
    all_scores = {**emotion_scores, **education_scores}
    fig = plot_spider_chart(all_scores)

    # Update table
    data = fetch_data()
    df = pd.DataFrame(data, columns=["ID", "Text", "Timestamp", "Scores"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["Scores"] = df["Scores"].apply(json.dumps)
    df = df.sort_values(by="Timestamp", ascending=False)

    return fig, df

def update_table(order: str):
    """
    Update table based on sort order
    """
    data = fetch_data()
    df = pd.DataFrame(data, columns=["ID", "Text", "Timestamp", "Scores"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["Scores"] = df["Scores"].apply(json.dumps)
    df = df.sort_values(by="Timestamp", ascending=(order == "Ascending"))
    return df

def get_entry_by_id(text_id: int):
    """
    Fetch a specific entry by ID
    """
    data = fetch_data()
    return next((entry for entry in data if entry[0] == text_id), None)

def display_graph_by_id(text_id: int):
    """
    Display spider chart for a specific text entry
    """
    entry = get_entry_by_id(text_id)
    if entry is None:
        return None, f"No entry found with ID {text_id}"

    items = {k: entry[3][k] for k in list(entry[3])[:4]}
    items["educational"] = entry[3]["LABEL_0"]
    fig = plot_spider_chart(items)
    return fig, f"Displaying graph for text: {entry[1][:100]}..."

with gr.Blocks() as app:
    gr.Markdown("<h1 style='font-size: 48px; text-align: left'>Text Analysis Dashboard</h1>")

    # New Analysis Section
    gr.Markdown("## Write the text you want to analyze")
    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(label="Enter text to analyze", lines=3 , value=text)
            analyze_btn = gr.Button("Analyze")

        with gr.Column(scale=2):
            new_plot_output = gr.Plot()
            # new_text_output = gr.Textbox(label="Analysis Info", interactive=False)

    gr.Markdown("---")  # Separator

    # Historical View Section
    gr.Markdown("## View Historical Data")
    with gr.Row():
        with gr.Column(scale=1):
            id_input = gr.Number(label="Enter Text ID to view graph", precision=0,minimum=1, value=1)
            view_btn = gr.Button("View Graph")

        with gr.Column(scale=2):
            historical_plot_output = gr.Plot()
            historical_text_output = gr.Textbox(label="Historical Info", interactive=False)

    gr.Markdown("---")  # Separator

    # Table Section
    gr.Markdown("## Data Table")
    with gr.Row():
        order_dropdown = gr.Dropdown(
            ["Ascending", "Descending"],
            label="Sort by Timestamp",
            value="Descending"
        )
    table_output = gr.Dataframe()

    # Set up event handlers
    analyze_btn.click(
        process_text,
        inputs=text_input,
        outputs=[new_plot_output, table_output]
    )

    view_btn.click(
        display_graph_by_id,
        inputs=id_input,
        outputs=[historical_plot_output, historical_text_output]
    )

    order_dropdown.change(
        update_table,
        inputs=order_dropdown,
        outputs=table_output
    )

# Launch the app
app.launch(server_name="0.0.0.0", server_port=7860)