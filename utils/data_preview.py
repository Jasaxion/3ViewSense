import gradio as gr
import json
import os
import base64
import math

APP_TITLE = "Data Preview Tool"
APP_DESC = 'Visualize and analyze data. Each line of the .jsonl should contain fields such as: {"images": [...], "problem": "...", "answer": "...", "model_generation": "..."}'

CUSTOM_CSS = r"""
:root {
  --card-bg: var(--block-background-fill);
  --muted: rgba(0,0,0,0.55);
  --shadow: 0 6px 22px rgba(0,0,0,0.08);
}
.dark, [data-theme="dark"] {
  --muted: rgba(255,255,255,0.6);
}
.app-title h1 { font-weight: 700; letter-spacing: -0.2px; margin: 0 0 6px 0; }
.app-desc { margin-top: -2px; color: var(--muted); }
.page-info { font-size: 0.95rem; opacity: 0.8; text-align: center; margin-top: 10px; }

.card {
  position: relative;
  background: var(--card-bg);
  border-radius: 18px;
  padding: 16px;
  box-shadow: var(--shadow);
  border: 1px solid rgba(0,0,0,0.06);
  margin: 16px 0;
}
.dark .card {
  border-color: rgba(255,255,255,0.12);
  background-color: #1f2937;
  color: #f3f4f6;
}
.index-pill { position: absolute; top: 12px; right: 14px; font-size: 12px; opacity: 0.6; }
.sec { margin-top: 8px; }
.sec-title { font-weight: 700; font-size: 14px; opacity: 0.8; margin-bottom: 6px; }

.md-wrap :is(h3, h4, h5) { margin: 6px 0 8px 0; }
.md-wrap p { margin: 0.4em 0; }

.images-wrap { display: flex; flex-wrap: wrap; gap: 10px; }
.images-wrap img {
  max-width: 400px;
  height: auto;
  border-radius: 12px;
  border: 1px solid rgba(0,0,0,0.05);
  box-shadow: 0 3px 12px rgba(0,0,0,0.07);
}

.think {
  color: #666;
  background: #f8f9fa;
  border-left: 3px solid #999;
  padding: 8px;
  margin-bottom: 8px;
  font-style: italic;
}

.answer-badge {
  display: inline-block;
  padding: 8px 14px;
  border-radius: 10px;
  font-weight: 700;
  background: rgba(16,185,129,0.12);
  font-size: 1.1em;
  margin: 0;
}
"""

def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            ext = os.path.splitext(image_path)[1].lower()
            mime_type = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.gif': 'image/gif', '.svg': 'image/svg+xml'}.get(ext, 'image/png')
            return f"data:{mime_type};base64,{encoded_string}"
    except Exception as e:
        print(f"Warning: cannot process image {image_path}: {e}")
        return None

def load_and_process_data(file_obj, file_path_text, choice):
    filepath = None
    if choice == "Upload File":
        if file_obj is None: raise gr.Error("Please upload a .jsonl file first.")
        filepath = file_obj.name
    else:
        if not file_path_text: raise gr.Error("Please input the local file path.")
        filepath = file_path_text.strip()

    if not os.path.exists(filepath):
        raise gr.Error("Invalid file path or file does not exist.")

    processed_data = []
    base_dir = os.path.dirname(filepath)

    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                record = json.loads(line)
                image_uris = []
                if "images" in record and isinstance(record["images"], list):
                    for img_rel_path in record["images"]:
                        img_abs_path = os.path.join(base_dir, img_rel_path)
                        img_uri = image_to_base64(img_abs_path)
                        if img_uri:
                            image_uris.append(img_uri)
                record['processed_images'] = image_uris
                processed_data.append(record)
            except json.JSONDecodeError:
                print(f"Warning: skip the line {i+1} with format error.")
                continue

    if not processed_data:
        raise gr.Error("File is empty or all lines cannot be parsed.")

    return processed_data, 1, "Data loaded successfully!"

def build_item_markdown(item, index):
    problem = str(item.get('problem', '')).replace("<image>", "").strip()
    solution = item.get('model_generation', '_No solution_')
    solution = solution.replace("<think>", "<div class='think'>").replace("</think>", "</div>")
    answer = str(item.get('answer', '_No answer_'))
    has_delimiters = '$' in answer or r'\(' in answer or r'\[' in answer
    if answer and not has_delimiters:
        answer = f"$${answer}$$"
    
    images = item.get('processed_images', [])

    images_html = ""
    if images:
        images_html += '<div class="images-wrap">'
        for uri in images:
            images_html += f'<img src="{uri}" alt="image" loading="lazy" />'
        images_html += '</div>'

    md = f"""
<div class="card">
<div class="index-pill">#{index + 1}</div>
<div class="sec sec-problem">
  <div class="sec-title">Problem</div>
  <div class="md-wrap">{problem}</div>
</div>
"""
    if images:
        md += f"""
<div class="sec sec-images">
  <div class="sec-title">Images</div>
  {images_html}
</div>
"""
    md += f"""
<div class="sec sec-solution">
  <div class="sec-title">Solution/Model Output</div>
  <div class="md-wrap">{solution}</div>
</div>
<div class="sec sec-answer">
  <div class="sec-title">Answer</div>
  <div class="answer-badge">{answer}</div>
</div>
</div>
"""
    return md

def update_display(data, page_num_str, items_per_page_str):
    if not data:
        return "> No data loaded", "Page 0 / 0 · Total 0", 1

    page_num = int(page_num_str)
    items_per_page = int(items_per_page_str)
    
    total_items = len(data)
    total_pages = math.ceil(total_items / items_per_page)
    page_num = max(1, min(page_num, total_pages))

    start_index = (page_num - 1) * items_per_page
    end_index = start_index + items_per_page
    page_data = data[start_index:end_index]

    display_md = "\n\n".join([build_item_markdown(item, start_index + i) for i, item in enumerate(page_data)])
    page_info_text = f"Page {page_num} / {total_pages} · Total {total_items}"
    
    return display_md, page_info_text, page_num


with gr.Blocks(theme=gr.themes.Soft(), title=APP_TITLE, css=CUSTOM_CSS) as demo:
    all_data = gr.State([])
    current_page = gr.State(1)

    gr.Markdown(f"# {APP_TITLE}", elem_classes=["app-title"])
    gr.Markdown(APP_DESC, elem_classes=["app-desc"])
    
    with gr.Row(variant="panel"):
        with gr.Column(scale=2):
            data_source_choice = gr.Radio(["Upload File", "Local Path"], label="Data Source", value="Upload File", interactive=True)
            file_input = gr.File(label="Upload .jsonl file", file_types=[".jsonl"], visible=True)
            path_input = gr.Textbox(label="Input the local .jsonl file path", placeholder="/path/to/your/file.jsonl", visible=False)
            load_btn = gr.Button("🚀 Load and Display Data", variant="primary")
            status_message = gr.Markdown("")
        with gr.Column(scale=3):
            items_per_page = gr.Radio(['1', '5', '10'], label="Items per page", value='5', interactive=True)
            with gr.Row():
                prev_btn = gr.Button("⬅️ Previous Page")
                next_btn = gr.Button("➡️ Next Page")
            with gr.Row():
                jump_page_input = gr.Number(label="Jump to", value=1, precision=0, interactive=True)
                jump_btn = gr.Button("Jump", variant="secondary")

    page_info = gr.Markdown("Page 0 / 0 · Total 0", elem_classes=["page-info"])
    display_output = gr.Markdown(
        "> Welcome! Load a .jsonl file above to begin.",
        latex_delimiters=[{"left": "$$", "right": "$$", "display": True}, {"left": "$", "right": "$", "display": False}, {"left": "\\(", "right": "\\)", "display": False}, {"left": "\\[", "right": "\\]", "display": True}]
    )

    def on_source_change(choice):
        return gr.update(visible=(choice == "Upload File")), gr.update(visible=(choice == "Local Path"))
    data_source_choice.change(on_source_change, inputs=[data_source_choice], outputs=[file_input, path_input])

    load_btn.click(
        fn=load_and_process_data,
        inputs=[file_input, path_input, data_source_choice],
        outputs=[all_data, current_page, status_message]
    ).then(
        fn=update_display,
        inputs=[all_data, current_page, items_per_page],
        outputs=[display_output, page_info, jump_page_input]
    )

    def change_page_fn(data, current_page_num, items_per_page_str, direction):
        page = int(current_page_num) + direction
        return str(page)

    prev_btn.click(change_page_fn, [all_data, current_page, items_per_page, gr.State(-1)], [current_page])
    next_btn.click(change_page_fn, [all_data, current_page, items_per_page, gr.State(1)], [current_page])

    def jump_page_fn(jump_to_page):
        return str(jump_to_page)
    jump_btn.click(jump_page_fn, [jump_page_input], [current_page])

    current_page.change(update_display, [all_data, current_page, items_per_page], [display_output, page_info, jump_page_input])
    items_per_page.change(lambda: "1", None, [current_page])


if __name__ == "__main__":
    demo.launch(share=False)