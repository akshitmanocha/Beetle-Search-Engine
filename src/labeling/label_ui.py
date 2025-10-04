#!/usr/bin/env python3
"""
Gradio Labeling UI - Manual labeling interface for blog classification.
"""

import csv
import json
import gradio as gr
import pandas as pd
from pathlib import Path
from datetime import datetime


class LabelingSession:
    def __init__(self):
        self.weak_csv = Path("data/labels/weak.csv")
        self.manual_csv = Path("data/labels/manual.csv")
        self.parsed_dir = Path("data/parsed")

        # Load weak labels
        self.weak_df = pd.read_csv(self.weak_csv)

        # Load or create manual labels
        if self.manual_csv.exists():
            self.manual_df = pd.read_csv(self.manual_csv)
            self.labeled_ids = set(self.manual_df['id'].values)
        else:
            self.manual_df = pd.DataFrame(columns=[
                'id', 'url', 'title', 'label', 'labeler_confidence',
                'notes', 'labeled_at'
            ])
            self.labeled_ids = set()

        # Prepare samples for labeling (prioritize uncertain, then low confidence)
        self.prepare_samples()
        self.current_index = 0

    def prepare_samples(self):
        """Prepare and prioritize samples for labeling."""
        # Filter out already labeled
        unlabeled = self.weak_df[~self.weak_df['id'].isin(self.labeled_ids)].copy()

        # Sort by: uncertain first, then by confidence (ascending)
        unlabeled['priority'] = unlabeled.apply(
            lambda row: 0 if row['label'] == 'uncertain' else row['confidence'],
            axis=1
        )
        unlabeled = unlabeled.sort_values('priority')

        self.samples = unlabeled.to_dict('records')

    def get_current_sample(self):
        """Get current sample for labeling."""
        if self.current_index >= len(self.samples):
            return None, "All samples labeled! 🎉"

        sample = self.samples[self.current_index]

        # Load full parsed data
        parsed_file = self.parsed_dir / f"{sample['id']}.json"
        with open(parsed_file, 'r') as f:
            data = json.load(f)

        return sample, data

    def format_display(self, sample, data):
        """Format sample for display in UI."""
        # Header
        output = f"### Sample {self.current_index + 1} / {len(self.samples)}\n"
        output += f"**Progress:** {len(self.labeled_ids)} labeled, {len(self.samples) - self.current_index} remaining\n\n"
        output += "---\n\n"

        # Title and URL
        output += f"# {data['title']}\n\n"
        output += f"**URL:** [{sample['url']}]({sample['url']})\n\n"

        # Metadata
        output += "### Metadata\n"
        output += f"- **Word Count:** {data['word_count']}\n"
        output += f"- **Authors:** {', '.join(data.get('authors', [])) or 'None'}\n"
        output += f"- **Publish Date:** {data.get('publish_date') or 'None'}\n"
        output += f"- **Code Blocks:** {data['code_blocks_count']}\n"
        output += f"- **First-Person Ratio:** {data['first_person_ratio']:.4f}\n"
        output += f"- **Has Author Bio:** {'Yes' if data['has_author_bio'] else 'No'}\n\n"

        # Weak label info
        output += "### Weak Label Prediction\n"
        output += f"- **Label:** {sample['label'].upper()}\n"
        output += f"- **Confidence:** {sample['confidence']:.2f}\n"
        output += f"- **Score:** {sample['score']}\n"
        output += f"- **Reasoning:** {sample['reasoning']}\n\n"

        # Content preview
        output += "### Content Preview (first 800 chars)\n"
        body_preview = data['body_text'][:800]
        if len(data['body_text']) > 800:
            body_preview += "..."
        output += f"```\n{body_preview}\n```\n\n"

        # Headers
        if data['headers']:
            output += "### Document Structure (Headers)\n"
            for i, h in enumerate(data['headers'][:10], 1):
                indent = "  " * (h['level'] - 1)
                output += f"{indent}{'#' * h['level']} {h['text']}\n"
            if len(data['headers']) > 10:
                output += f"\n... and {len(data['headers']) - 10} more headers\n"

        return output

    def save_label(self, label, confidence, notes):
        """Save a manual label."""
        if self.current_index >= len(self.samples):
            return "No more samples to label!", self.format_stats()

        sample = self.samples[self.current_index]

        # Add to manual labels
        new_label = {
            'id': sample['id'],
            'url': sample['url'],
            'title': sample['title'],
            'label': label,
            'labeler_confidence': confidence,
            'notes': notes,
            'labeled_at': datetime.now().isoformat()
        }

        self.manual_df = pd.concat([
            self.manual_df,
            pd.DataFrame([new_label])
        ], ignore_index=True)

        self.labeled_ids.add(sample['id'])

        # Save to CSV
        self.manual_df.to_csv(self.manual_csv, index=False)

        # Move to next sample
        self.current_index += 1

        return self.get_next_display()

    def skip_sample(self):
        """Skip current sample."""
        self.current_index += 1
        return self.get_next_display()

    def get_next_display(self):
        """Get next sample display."""
        sample, data = self.get_current_sample()

        if sample is None:
            return data, self.format_stats()  # data contains completion message

        display = self.format_display(sample, data)
        stats = self.format_stats()

        return display, stats

    def format_stats(self):
        """Format labeling statistics."""
        if len(self.manual_df) == 0:
            return "No labels yet."

        stats = f"### Labeling Statistics\n"
        stats += f"**Total Labeled:** {len(self.manual_df)}\n\n"

        label_counts = self.manual_df['label'].value_counts()
        for label, count in label_counts.items():
            pct = 100 * count / len(self.manual_df)
            stats += f"- **{label}:** {count} ({pct:.1f}%)\n"

        stats += f"\n**Last labeled:** {self.manual_df.iloc[-1]['labeled_at']}\n"

        return stats


# Initialize session
session = LabelingSession()


def initial_load():
    """Load first sample."""
    return session.get_next_display()


def label_blog(confidence, notes):
    """Label as blog."""
    return session.save_label('blog', confidence, notes)


def label_not_blog(confidence, notes):
    """Label as not-blog."""
    return session.save_label('not-blog', confidence, notes)


def skip():
    """Skip sample."""
    return session.skip_sample()


# Build Gradio UI
with gr.Blocks(title="Blog Labeling Interface", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📝 Blog Labeling Interface")
    gr.Markdown("Label whether each page is a blog post or not. Focus on content quality and structure.")

    with gr.Row():
        with gr.Column(scale=3):
            sample_display = gr.Markdown(value="Loading...")

        with gr.Column(scale=1):
            gr.Markdown("## 🏷️ Label This Sample")

            confidence = gr.Slider(
                minimum=1,
                maximum=5,
                step=1,
                value=3,
                label="Your Confidence (1=unsure, 5=very confident)",
                info="How confident are you in this label?"
            )

            notes = gr.Textbox(
                label="Notes (optional)",
                placeholder="Any observations or reasons for your decision...",
                lines=3
            )

            with gr.Row():
                blog_btn = gr.Button("✅ BLOG", variant="primary", size="lg")
                not_blog_btn = gr.Button("❌ NOT BLOG", variant="secondary", size="lg")

            skip_btn = gr.Button("⏭️ Skip", variant="stop", size="sm")

            gr.Markdown("---")
            stats_display = gr.Markdown(value="Loading stats...")

    # Event handlers
    blog_btn.click(
        fn=label_blog,
        inputs=[confidence, notes],
        outputs=[sample_display, stats_display]
    ).then(
        fn=lambda: ("", 3),  # Reset notes and confidence
        outputs=[notes, confidence]
    )

    not_blog_btn.click(
        fn=label_not_blog,
        inputs=[confidence, notes],
        outputs=[sample_display, stats_display]
    ).then(
        fn=lambda: ("", 3),
        outputs=[notes, confidence]
    )

    skip_btn.click(
        fn=skip,
        outputs=[sample_display, stats_display]
    ).then(
        fn=lambda: ("", 3),
        outputs=[notes, confidence]
    )

    # Load initial sample
    demo.load(
        fn=initial_load,
        outputs=[sample_display, stats_display]
    )


if __name__ == "__main__":
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
