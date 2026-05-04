# Sprint 2 Web Application Architecture

This document outlines the architecture, design choices, and deployment strategy for the ISM CyberRAG web interface developed during Sprint 2. It also documents our roadmap for Sprint 3.

## 1. Technical Stack

**Backend: FastAPI**
- The web backend is powered by FastAPI, serving as a lightweight orchestration layer between the user interface and our RAG pipeline.
- It exposes a single `/chat` REST endpoint that receives the user query, generates embeddings locally (`nomic-embed-text-v1.5`), retrieves chunks from Supabase, reranks them using `ms-marco-MiniLM-L-6-v2`, and generates an answer via Groq (Llama 3.1).

**Frontend: Vanilla Web Components**
- **HTML structure**: Rendered using Jinja2 templates (`app/templates/index.html`).
- **Styling**: Vanilla CSS (`app/static/style.css`).
- **Logic**: Vanilla JavaScript utilizing the browser's native `fetch` API.

> **Design Rationale**: We deliberately avoided complex frontend frameworks (like React or Vue) to keep the repository lightweight and easily deployable as a single container, while maintaining extremely quick load times.

## 2. Sprint 2 UI Innovations (The "Enterprise Knowledge Base" Design)

To provide maximum transparency and instill authority, we implemented an official **Australian Government (ASD/ACSC)** inspired design architecture:

1. **Top Identity Bar**: Emulates the official cyber.gov.au structure featuring standard utility navigation and the ASD/ACSC agency title blocks.
2. **Hero Search Banner**: A definitive, high-contrast Navy Blue (`#032049`) hero section focusing purely on querying the Information Security Manual.
3. **Left-Hand Side (LHS) [Search & Results]**: 
   - A traditional, formalized search input triggering the hybrid retrieval.
   - The LLM's generated response appears cleanly below the search box, stepping away from colloquial "chat bubbles" toward authoritative system answers.
4. **Right-Hand Side (RHS) [Reference Documents]**: 
   - A dedicated pane titled "Reference Documents". 
   - As the LLM responds, this pane dynamically populates with the raw context chunks retrieved from the ISM.
   - **Metrics Visibility**: Each chunk displays its metadata tags (`control_id`, `category`) and its exact **Rerank Score** for full transparency.

**Aesthetics**:
The app strictly utilizes a high-contrast accessibility-friendly color palette (Navy, White, Light Grey) with standard system sans-serif fonts, reflecting an official government portal.

## 3. Deployment Infrastructure (Hugging Face Spaces)

**Why Hugging Face Spaces?**
We deploy to HF Spaces because the free tier provides 16GB of CPU RAM, which is sufficient to load our Embedding and Reranking models completely into system memory, avoiding network latency during document retrieval.

**Docker Strategy:**
- We utilize a custom `Dockerfile` rather than HF's default Gradio abstractions.
- We restrict `torch` installations to the CPU-only wheels via `--index-url https://download.pytorch.org/whl/cpu`, dramatically lowering the image size and build times.
- **Warm Booting**: SentanceTransformers models (`nomic` and `ms-marco`) are forcibly downloaded inside the Docker `RUN` command. This ensures that when the HF Space spins up from a "sleeping" state, the models are instantly available on disk, preventing the user from waiting for a 2GB model download on their first query.

## 4. Sprint 3 Roadmap (Web App Enhancements)

While Sprint 2 solidifies the Chat UI and the deployment platform, Sprint 3 will expand the application's scope:

### The Analytics Dashboard
Currently, all RAGAS evaluation charts (Latency, Faithfulness, Category Breakdowns) are generated offline inside `notebooks/sprint2_development.ipynb`.

**Plan for Sprint 3:**
1. Generate the final Sprint 3 evaluation benchmark charts mapping our advanced upgrades (e.g., memory, agents).
2. Activate the `/evaluations` route within the FastAPI web app.
3. Expose these static visualizations in a dedicated, public-facing Analytics Dashboard, accessible from the main top navigation bar.

> *Note: For the Sprint 2 deployment, the "Evaluations" link in the top navigation has been deliberately hidden. Building out the evaluations page is deferred to Sprint 3 so that only the final, most advanced RAG metrics are presented to stakeholders.*
