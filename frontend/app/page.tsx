"use client";

import { ChangeEvent, FormEvent, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Page = { page: number; text: string; source: "text_layer" | "vlm_ocr" };
type Citation = { page: number; excerpt: string };
type Document = { id: string; name: string; pages: Page[]; text_layer_pages: number; ocr_pages: number };
type Message = { role: "user" | "assistant"; content: string; citations?: Citation[] };

async function readError(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: string };
    return body.detail ?? "Something went wrong";
  } catch {
    return "Something went wrong";
  }
}

type Source = "file" | "url";

export default function Home() {
  const [document, setDocument] = useState<Document | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [source, setSource] = useState<Source>("file");
  const [url, setUrl] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  function chooseFile(event: ChangeEvent<HTMLInputElement>) {
    const selected = event.target.files?.[0] ?? null;
    setFile(selected);
    setDocument(null);
    setMessages([]);
    setError("");
  }

  function chooseSource(next: Source) {
    setSource(next);
    setDocument(null);
    setMessages([]);
    setError("");
  }

  async function upload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file) return;
    setLoading(true);
    setError("");
    const data = new FormData();
    data.append("file", file);
    try {
      const response = await fetch(`${API_URL}/api/documents`, { method: "POST", body: data });
      if (!response.ok) throw new Error(await readError(response));
      setDocument((await response.json()) as Document);
      setMessages([]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not upload the PDF");
    } finally {
      setLoading(false);
    }
  }

  async function loadFromUrl(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = url.trim();
    if (!trimmed) return;
    setLoading(true);
    setError("");
    try {
      const response = await fetch(`${API_URL}/api/documents/from-url`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: trimmed }),
      });
      if (!response.ok) throw new Error(await readError(response));
      setDocument((await response.json()) as Document);
      setMessages([]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load the PDF from that URL");
    } finally {
      setLoading(false);
    }
  }

  async function ask(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = question.trim();
    if (!document || !trimmed || loading) return;
    setQuestion("");
    setMessages((current) => [...current, { role: "user", content: trimmed }]);
    setLoading(true);
    setError("");
    try {
      const response = await fetch(`${API_URL}/api/documents/${document.id}/questions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmed }),
      });
      if (!response.ok) throw new Error(await readError(response));
      const answer = (await response.json()) as { answer: string; citations: Citation[] };
      setMessages((current) => [...current, { role: "assistant", content: answer.answer, citations: answer.citations }]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not answer the question");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="shell">
      <header className="hero">
        <div className="eyebrow">Grounded document intelligence</div>
        <h1>Ask your PDF anything.</h1>
        <p>Upload a document, let the agent extract its text, and get answers with page-level citations.</p>
      </header>

      <section className="workspace">
        <aside className="upload-panel">
          <div className="section-label">01 · Upload</div>
          <h2>Choose a PDF</h2>
          <p className="muted">Text pages are extracted directly. Scanned pages are read with vision OCR.</p>

          <div className="source-toggle" role="tablist" aria-label="PDF source">
            <button
              type="button"
              role="tab"
              aria-selected={source === "file"}
              className={source === "file" ? "active" : ""}
              onClick={() => chooseSource("file")}
            >
              Upload file
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={source === "url"}
              className={source === "url" ? "active" : ""}
              onClick={() => chooseSource("url")}
            >
              Paste URL
            </button>
          </div>

          {source === "file" ? (
            <form onSubmit={upload}>
              <label className="dropzone">
                <input type="file" accept="application/pdf,.pdf" onChange={chooseFile} />
                <span className="file-icon">↑</span>
                <strong>{file ? file.name : "Drop a PDF here"}</strong>
                <span>{file ? `${(file.size / 1024 / 1024).toFixed(1)} MB selected` : "or click to browse · max 25 MB"}</span>
              </label>
              <button className="primary-button" disabled={!file || loading} type="submit">
                {loading && !document ? "Parsing…" : "Parse document"}
              </button>
            </form>
          ) : (
            <form className="url-form" onSubmit={loadFromUrl}>
              <label className="url-field">
                <span>PDF URL</span>
                <input
                  type="url"
                  inputMode="url"
                  placeholder="https://example.com/report.pdf"
                  value={url}
                  onChange={(event) => setUrl(event.target.value)}
                />
              </label>
              <p className="muted small">The server fetches the file directly — only public http(s) links to a PDF are supported.</p>
              <button className="primary-button" disabled={!url.trim() || loading} type="submit">
                {loading && !document ? "Fetching…" : "Load from URL"}
              </button>
            </form>
          )}

          {document && (
            <div className="document-card">
              <div className="document-title">✓ {document.name}</div>
              <div className="stats">
                <span><b>{document.pages.length}</b> pages</span>
                <span><b>{document.text_layer_pages}</b> text</span>
                <span><b>{document.ocr_pages}</b> OCR</span>
              </div>
            </div>
          )}
          {error && <div className="error">{error}</div>}
        </aside>

        <section className="chat-panel">
          <div className="section-label">02 · Ask questions</div>
          <h2>Conversation</h2>
          {!document ? (
            <div className="empty-state"><span>✦</span><p>Your answers will appear here<br />after you parse a document.</p></div>
          ) : (
            <>
              <div className="messages">
                {messages.length === 0 && <div className="empty-state compact"><span>✦</span><p>What would you like to know about this document?</p></div>}
                {messages.map((message, index) => (
                  <div className={`message ${message.role}`} key={`${message.role}-${index}`}>
                    <div className="message-role">{message.role === "user" ? "You" : "Agent"}</div>
                    <div className="message-content">{message.content}</div>
                    {message.citations?.map((citation, citationIndex) => (
                      <div className="citation" key={`${citation.page}-${citationIndex}`}><b>PAGE {citation.page}</b><br />“{citation.excerpt}”</div>
                    ))}
                  </div>
                ))}
                {loading && document && <div className="thinking">Searching the document<span>•••</span></div>}
              </div>
              <form className="question-form" onSubmit={ask}>
                <input value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask a question about this document…" />
                <button type="submit" disabled={!question.trim() || loading} aria-label="Send question">→</button>
              </form>
            </>
          )}
        </section>
      </section>

      {document && (
        <section className="preview-panel">
          <div className="section-label">03 · Source</div>
          <h2>Extracted text</h2>
          <p className="muted">This is the source context used by the agent.</p>
          <div className="page-list">
            {document.pages.map((page) => (
              <details key={page.page}>
                <summary>Page {page.page}<span>{page.source === "text_layer" ? "Text layer" : "VLM OCR"}</span></summary>
                <div className="page-text">{page.text}</div>
              </details>
            ))}
          </div>
        </section>
      )}
    </main>
  );
}
