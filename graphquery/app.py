"""
app.py — Gradio UI for the RAG Agent
Talks to the FastAPI backend (api.py) via HTTP + SSE streaming.
Dark terminal aesthetic, streaming chat, loading states.
"""

from __future__ import annotations

import httpx
import json
import gradio as gr

from config import settings

API_BASE = f"http://localhost:{settings.api_port}"


# ── HTTP helpers ───────────────────────────────────────────────────

def api_upload(files) -> tuple[str, str]:
    """POST files to /upload and return (status_msg, file_list_md)."""
    if not files:
        return "⚠️  No files selected.", "_No documents loaded._"

    try:
        with httpx.Client(timeout=120.0) as client:
            multipart = [
                ("files", (f.name.split("/")[-1], open(f.name, "rb"),
                 "application/octet-stream"))
                for f in files
            ]
            resp = client.post(f"{API_BASE}/upload", files=multipart)
            resp.raise_for_status()
            data = resp.json()

        names = "\n".join(f"- `{n}`" for n in data["files"])
        return (
            f"✅  {data['message']}",
            f"**Loaded files:**\n{names}",
        )
    except httpx.HTTPStatusError as e:
        detail = e.response.json().get("detail", str(e))
        return f"❌  {detail}", "_No documents loaded._"
    except Exception as e:
        return f"❌  {e}", "_No documents loaded._"


def api_clear() -> tuple[str, str]:
    """DELETE /documents and return (status_msg, file_list_md)."""
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.delete(f"{API_BASE}/documents")
            resp.raise_for_status()
        return "🗑️  All documents cleared.", "_No documents loaded._"
    except Exception as e:
        return f"❌  {e}", "_No documents loaded._"


def _text(content) -> str:
    """Extract plain text from Gradio 6.x content.
    Content can be a plain string OR a list of {'text': ..., 'type': 'text'} dicts.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part["text"] if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def stream_respond(message: str, history: list):
    """
    Generator consumed by Gradio's streaming chatbot.
    Calls /chat SSE endpoint and yields the accumulating answer.
    Compatible with Gradio 6.x dict-based message format where
    content may be a list of {'text': ..., 'type': 'text'} objects.
    """
    if not message.strip():
        yield history, ""
        return

    accumulated = ""

    # Build API history from previous completed turns.
    # history is a flat list: [user_msg, assistant_msg, user_msg, ...]
    # Each entry is a dict with 'role' and 'content' (content may be str or list).
    api_history = [
        [_text(history[i]["content"]), _text(history[i + 1]["content"])]
        for i in range(0, len(history) - 1, 2)
        if i + 1 < len(history)
    ]
    payload = {"question": message, "history": api_history}

    # Append the new turn for display (plain strings for new messages)
    history = history + [{"role": "user", "content": message},
                         {"role": "assistant", "content": ""}]

    try:
        with httpx.Client(timeout=120.0) as client:
            with client.stream("POST", f"{API_BASE}/chat", json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    if line.startswith("data:"):
                        raw = line[5:].strip()
                        if not raw:
                            continue
                        try:
                            event = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        if "token" in event:
                            accumulated += event["token"]
                            history[-1]["content"] = accumulated
                            yield history, ""
                        elif "done" in event:
                            break
                        elif "error" in event:
                            history[-1]["content"] = f"⚠️  {event['error']}"
                            yield history, ""
                            return
    except Exception as e:
        history[-1]["content"] = f"❌  Could not reach API server: {e}"
        yield history, ""



# ── Custom CSS (dark terminal aesthetic) ──────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Sora:wght@300;500;700&display=swap');

:root {
    --bg:          #0d0f14;
    --surface:     #161920;
    --border:      #2a2d38;
    --accent:      #7ee8a2;
    --accent2:     #56cfe1;
    --text:        #e2e8f0;
    --muted:       #6b7280;
    --user-bubble: #1e293b;
    --bot-bubble:  #14211a;
    --radius:      10px;
}

body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'Sora', sans-serif !important;
    color: var(--text) !important;
}

#header {
    text-align: center;
    padding: 2rem 1rem 0.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
#header h1 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem;
    color: var(--accent);
    letter-spacing: -0.5px;
    margin: 0;
}
#header p {
    color: var(--muted);
    font-size: 0.85rem;
    margin: 0.4rem 0 0;
    font-family: 'JetBrains Mono', monospace;
}

.panel {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1rem !important;
}

button.primary {
    background: var(--accent) !important;
    color: #0d0f14 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 6px !important;
}
button.secondary {
    background: transparent !important;
    color: var(--accent2) !important;
    border: 1px solid var(--accent2) !important;
    font-family: 'JetBrains Mono', monospace !important;
    border-radius: 6px !important;
}
button:hover { opacity: 0.85 !important; }

.chatbot {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}
.chatbot .message.user {
    background: var(--user-bubble) !important;
    border-radius: 8px 8px 2px 8px !important;
    color: var(--text) !important;
    font-family: 'Sora', sans-serif !important;
}
.chatbot .message.bot {
    background: var(--bot-bubble) !important;
    border-left: 3px solid var(--accent) !important;
    border-radius: 2px 8px 8px 8px !important;
    color: var(--text) !important;
    font-family: 'Sora', sans-serif !important;
}

textarea, input[type="text"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
    border-radius: 6px !important;
}
textarea:focus, input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(126,232,162,0.15) !important;
}

.status-box {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: var(--accent);
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    padding: 0.5rem !important;
}
.file-list {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--muted) !important;
}
label, .label-wrap {
    color: var(--muted) !important;
    font-size: 0.78rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}
.upload-button {
    border: 1.5px dashed var(--border) !important;
    background: transparent !important;
    color: var(--muted) !important;
    border-radius: var(--radius) !important;
}
.upload-button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}
"""


# ── Layout ─────────────────────────────────────────────────────────

with gr.Blocks(title="RAG Agent · llama3") as demo:

    gr.HTML("""
    <div id="header">
        <h1>⬡ RAG Agent</h1>
        <p>LangGraph · Qdrant · Ollama llama3 · Hybrid Search · Local &amp; Private</p>
    </div>
    """)

    with gr.Row():

        # ── Left sidebar ──────────────────────────────────────────
        with gr.Column(scale=1, min_width=280):

            gr.Markdown("### 📂 Documents", elem_classes=["panel"])

            file_upload = gr.File(
                label="Upload PDF / TXT / MD",
                file_count="multiple",
                file_types=[".pdf", ".txt", ".md"],
            )

            with gr.Row():
                btn_upload = gr.Button("⬆ Index Files", variant="primary")
                btn_clear  = gr.Button("🗑 Clear", variant="secondary")

            status_box = gr.Markdown(
                "_Upload documents to begin._",
                elem_classes=["status-box"]
            )
            file_list = gr.Markdown(
                "_No documents loaded._",
                elem_classes=["file-list"]
            )

            gr.Markdown("""---
**Model:** `llama3` via Ollama
**Embedding:** `nomic-embed-text`
**Vector store:** Qdrant (persistent)
**Retrieval:** BM25 + Dense + Rerank
**Framework:** LangGraph + FastAPI
""", elem_classes=["file-list"])

        # ── Chat area ─────────────────────────────────────────────
        with gr.Column(scale=3):

            chatbot = gr.Chatbot(
                label="",
                height=520,
                show_label=False,
                avatar_images=(None, "🤖"),
            )

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask anything about your documents…",
                    show_label=False,
                    scale=8,
                    lines=1,
                )
                send_btn = gr.Button("Send →", variant="primary", scale=1)

            gr.Examples(
                examples=[
                    "Summarise the main topics in these documents.",
                    "What are the key findings?",
                    "List any dates or deadlines mentioned.",
                    "Explain the most important concept in simple terms.",
                ],
                inputs=msg_input,
                label="Quick prompts",
            )

    # ── Wiring ─────────────────────────────────────────────────────

    btn_upload.click(
        api_upload,
        inputs=[file_upload],
        outputs=[status_box, file_list],
    )

    btn_clear.click(
        api_clear,
        inputs=[],
        outputs=[status_box, file_list],
    ).then(lambda: [], outputs=[chatbot])  # empty list works for both formats

    send_btn.click(
        stream_respond,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, msg_input],
    )

    msg_input.submit(
        stream_respond,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, msg_input],
    )


if __name__ == "__main__":
    demo.launch(
        server_name=settings.gradio_host,
        server_port=settings.gradio_port,
        share=False,
        show_error=True,
        css=CSS,
    )
