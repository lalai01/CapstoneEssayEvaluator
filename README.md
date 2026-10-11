# AI Essay Evaluator

A web app that scores essays against a configurable rubric, gives written feedback, reads handwritten and printed essays with OCR, and lets teachers correct the AI so later evaluations improve.

- **Frontend:** React 19 + Vite + Tailwind, hosted on Firebase Hosting (`https://capstoneessayevaluator.web.app`)
- **Backend:** FastAPI (Python 3.11) in Docker on a Contabo VPS (`https://api.essay-evaluator.duckdns.org`)
- **Data and auth:** Supabase (Postgres, Auth with Google/Yahoo, Storage)
- **AI models:** DeepSeek API, Ollama (Llama 3.2 3B, Gemma 2 2B) and llama.cpp (Phi-3.5 Mini), all reachable from the backend

---

## Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Project structure](#project-structure)
4. [Running locally](#running-locally)
5. [Environment variables](#environment-variables)
6. [Deployment](#deployment)
7. [Database (Supabase)](#database-supabase)
8. [Rubrics and scoring](#rubrics-and-scoring)
9. [Prompt Playground statistics](#prompt-playground-statistics)
10. [OCR and handwriting](#ocr-and-handwriting)
11. [API reference](#api-reference)

---

## Features

| Area | What it does |
|---|---|
| **Evaluate Essay** | Paste text or upload an image, PDF or DOCX. Analytic mode scores every rubric criterion. Holistic mode gives one 1–5 score. Optional RAG adds feedback from similar past essays. Signed-in teachers choose which rubric to score with. |
| **Evaluation report** | **Generate Evaluation Report** creates a multi-page PDF. It includes student, class and teacher details, the rubric used, the total, percentage, weighted score and grade, a per-criterion breakdown (with override scores beside the AI's), strengths and focus areas, criterion comments, feedback, grammar fixes, recommendations, a teacher comment and signature area, and optionally the full essay. |
| **Satisfaction rating** | After an evaluation, the user rates it 1–10. The rating is stored with Knowledge Base entries and teacher overrides. |
| **Teacher override** | **Reject & Override** an evaluation. The teacher enters their own score per criterion, **teacher's remarks** (required, in their own words, with the AI feedback shown for reference) and **suggestions for the student** (optional). Overrides go to the Learning Hub, feed back into RAG, and the remarks fill the teacher comments in the evaluation report. |
| **Knowledge Base** | Evaluations the teacher accepted and saved. Each entry shows which AI scored it, its rubric, and whether the same essay was also overridden (with those remarks inline). Search, sort, filter by AI model, and filter by saved-only or also-overridden. |
| **Learning Hub** | Teacher overrides, with essay title, the AI that was overridden, AI-vs-teacher scores, remarks and suggestions. The same search, sort and AI filter, plus filters for "scores changed" and "also in Knowledge Base". |
| **AI Prompt Playground** | Runs the same prompt on three models side by side. A **Score Comparison** table (Category / Primary / Compare / Diff) compares any two of them and can be saved as a PDF report. Agreement statistics are described [below](#prompt-playground-statistics). |
| **Rubrics** (signed-in teachers) | Teachers create their own rubrics. Each criterion has its own maximum score (1–2 up to 1–10), weight and level descriptors, and teachers can add criteria of their own. A teacher's **default** rubric is used for their evaluations. They can copy the school-wide rubric as a starting point. Personal rubrics are private to their owner. |
| **School-wide rubric** (admin only) | Admins create and activate the school-wide rubric, the default for every teacher without their own. Teachers can view it but not change it; the backend rejects such changes with 403. |
| **Surveys** | Admins create surveys; logged-in users answer them in the **Surveys** tab. |
| **OCR Labeler** (admin) | Correct OCR output on uploaded images to build TrOCR fine-tuning data. |
| **Rate Us** | Public ratings with threaded comments and reactions. |

A user is an admin if their Supabase `user_metadata.role` or `app_metadata.role` is `admin`, or if their email is in `ADMIN_EMAILS` (backend) / `VITE_ADMIN_EMAILS` (frontend). Both lists default to `admin_essay_capstone@gmail.com,galangec24@gmail.com`; keep them identical. Admins sign in through **Sign In → Admin Login** (email and password) or with Google if their email is listed.

---

## Architecture

```
Browser (React, Firebase Hosting)
   │  Supabase Auth (Google / Yahoo) → JWT
   ▼
FastAPI backend (Docker, Contabo VPS)
   ├── evaluation_graph.py   LangGraph pipeline: validate → score → RAG → feedback
   ├── evaluator.py          rubric normalization, AI scoring, heuristic fallback, feedback
   ├── ai_models.py          DeepSeek / Ollama / llama.cpp clients (Prompt Playground)
   ├── ocr_utils.py, handwriting_ocr.py, ocr_jobs.py, image_preprocess.py
   │                         Tesseract for print, TrOCR for handwriting, background PDF jobs
   └── rag.py                TF-IDF retrieval over the knowledge base and teacher overrides
   │
   ├──► Supabase (tables, storage bucket "ocr-samples")
   ├──► Ollama   http://ollama:11434       (llama3.2:3b scoring/feedback, gemma2:2b)
   ├──► llama.cpp http://llamacpp:8080/v1   (Phi-3.5 Mini)
   └──► DeepSeek API                         (OCR correction, Playground)
```

**How an essay is evaluated** (`POST /evaluate`)
1. The text is checked: at least 20 words, English, and real words.
2. The active rubric is loaded (`get_active_rubric`) and normalized.
3. Llama 3.2 scores every criterion within its own range and returns per-criterion feedback, grammar issues and recommendations as JSON. If that fails, heuristic scores are used, rescaled to the rubric.
4. RAG context is optionally added, rule-based feedback is built, and Llama rewrites it into a supportive paragraph.
5. The response includes `scores.criteria_meta`, so the frontend shows the right labels, maximums and weights.

---

## Project structure

```
backend/
  main.py                 FastAPI app and all routes
  evaluation_graph.py     LangGraph evaluation pipeline
  evaluator.py            scoring, rubric handling, feedback, OCR text correction
  ai_models.py            model clients for the Prompt Playground
  auth.py                 Supabase JWT verification
  supabase_client.py      anon + service-role Supabase clients
  rag.py                  similar-essay retrieval
  ocr_utils.py            engine selection, DOCX/PDF extraction
  handwriting_ocr.py      TrOCR line segmentation and transcription
  ocr_jobs.py             background PDF OCR jobs
  image_preprocess.py     image cleanup before OCR
  export_training_data.py exports labeled OCR samples for fine-tuning
  train_trocr.py          fine-tunes TrOCR on Google Colab (T4)
  Dockerfile, requirements.txt

frontend/
  src/App.jsx             tabs and layout
  src/api.js              axios client (attaches the Supabase token)
  src/lib/scoring.js      rubric-aware score math shared by the UI
  src/lib/aiComparison.js Prompt Playground statistics
  src/components/         one component per tab, plus shared UI
  public/                 favicon, PWA manifest, service worker
  firebase.json           Firebase Hosting config (serves dist/)

.github/workflows/        build and deploy the frontend to Firebase
```

---

## Running locally

### Backend

Requirements: Python 3.11, Tesseract OCR, Java (for LanguageTool), and an Ollama instance with `llama3.2:3b` and `gemma2:2b` pulled.

```bash
cd backend
python -m venv venv
venv\Scripts\activate            # macOS/Linux: source venv/bin/activate
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
# create backend/.env (see Environment variables), then:
uvicorn main:app --reload --port 8000
```

`handwriting_ocr.py` loads TrOCR from `/opt/trocr`. The Dockerfile downloads it there. When you run outside Docker, either use Docker for handwriting OCR or save `microsoft/trocr-base-handwritten` to that path.

### Frontend

```bash
cd frontend
npm install
# create frontend/.env.local with VITE_API_URL=http://localhost:8000 plus the Supabase keys
npm run dev                       # http://localhost:5173
```

---

## Environment variables

### Backend (`backend/.env`, passed to the container)

| Variable | Required | Purpose |
|---|---|---|
| `SUPABASE_URL` | yes | Supabase project URL |
| `SUPABASE_KEY` | yes | Supabase anon key |
| `SUPABASE_SERVICE_KEY` | recommended | Service-role key for admin writes (rubrics, surveys) and reading the active rubric past RLS |
| `SUPABASE_JWT_SECRET` | recommended | Verifies user tokens. Without it, tokens are decoded **without** signature verification, which is fine for development only. |
| `DEEPSEEK_API_KEY` | for DeepSeek | OCR correction and the Playground DeepSeek slot |
| `OLLAMA_URL` | no | Default `http://ollama:11434` |
| `LLAMACPP_URL` | no | Default `http://llamacpp:8080/v1` |
| `LLAMACPP_MODEL` | no | Default `Phi-3.5-mini-instruct-Q4_K_M.gguf` |
| `ADMIN_EMAILS` | no | Comma-separated admin emails |
| `PDF_OCR_WORKERS` | no | Scanned PDF pages OCR'd at once. Default `min(4, CPU cores)` |
| `TROCR_QUANTIZE` | no | `1` runs TrOCR with int8 quantization: about 2× faster handwriting OCR, slightly less accurate. Off by default. |

### Frontend (`frontend/.env.production`, `frontend/.env.local`)

| Variable | Purpose |
|---|---|
| `VITE_API_URL` | Backend base URL |
| `VITE_SUPABASE_URL` | Supabase project URL |
| `VITE_SUPABASE_ANON_KEY` | Supabase anon key (safe to expose; protect data with RLS) |
| `VITE_ADMIN_EMAILS` | Optional comma-separated admin emails; must match the backend's `ADMIN_EMAILS` |

---

## Deployment

### Backend: Contabo VPS

The backend runs with Docker Compose on the VPS, next to the `ollama` and `llamacpp` services. Render is no longer used because it can't handle the model and OCR workload.

```bash
ssh <user>@<vps>
cd ~/CapstoneEssayEvaluator
git pull
docker compose build backend
docker compose up -d backend
docker compose logs -f backend    # check it started
```

To deploy a fine-tuned TrOCR model, copy `trocr-finetuned/` to `backend/trocr-finetuned/` on the VPS. Then replace the TrOCR download `RUN` block in `backend/Dockerfile` with `COPY trocr-finetuned /opt/trocr` and rebuild.

### Frontend: Firebase Hosting

Pushing to `main` triggers `.github/workflows/firebase-hosting-merge.yml`, which builds `frontend/` and deploys it. Pull requests get a preview channel. To deploy manually:

```bash
cd frontend
npm run build
firebase deploy --only hosting
```

The repository secret `FIREBASE_SERVICE_ACCOUNT_CAPSTONEESSAYEVALUATOR` must exist on GitHub.

---

## Database (Supabase)

| Table / object | Used for |
|---|---|
| `knowledge_base` | Saved evaluations: `title`, `essay`, fixed columns for the five built-in criteria, `scores` (jsonb, the full result for any rubric), `ai_model`, `criterion_feedback`, `satisfaction` 1–10 |
| `learning_feedback` | Teacher overrides: `title`, `ai_model`, `original_essay`, `original_scores` (jsonb), `teacher_scores` (jsonb), `teacher_feedback` (the teacher's remarks), `suggested_changes` (suggestions for the student), `satisfaction` |
| `rubrics` | `title`, `description`, `criteria` (jsonb, see below), `is_active`, `is_system` (school-wide vs personal), `user_id` (owner) |
| `saved_essays` | Playground drafts |
| `ratings`, `comments`, `comment_reactions` | Rate Us |
| `surveys`, `survey_questions`, `survey_responses` | Surveys |
| `ocr_training_data` | OCR samples and human corrections |
| Storage bucket `ocr-samples` | Uploaded OCR images |
| RPC `get_user_profiles` | Names and avatars for comments and ratings |

**One-time migrations.** Run these in the Supabase SQL editor before deploying this version:

```sql
-- Teacher-defined rubrics (required): separates school-wide rubrics from teachers' own.
alter table rubrics add column if not exists is_system boolean not null default false;
-- Existing rubrics were created by admins, so they become school-wide.
update rubrics set is_system = true;

-- Organizing the Knowledge Base and Learning Hub (recommended).
alter table knowledge_base    add column if not exists scores jsonb;
alter table knowledge_base    add column if not exists ai_model text;
alter table learning_feedback add column if not exists teacher_scores jsonb;
alter table learning_feedback add column if not exists title text;
alter table learning_feedback add column if not exists ai_model text;
```

- Without `rubrics.is_system`, evaluations still work and use the latest active rubric, but the Rubrics tab shows a "migration needed" error.
- Without the other columns, nothing is lost: saves drop the missing columns and keep their values inside an existing jsonb column (`knowledge_base.criterion_feedback` or `learning_feedback.original_scores`), and the app reads either location.
- Entries saved before this version have no AI model recorded, so they appear as "Unknown AI (older entry)".

---

## Rubrics and scoring

Rubrics are stored in `rubrics.criteria` in this shape:

```json
{
  "grammar":    { "label": "Grammar",    "max_score": 10, "weight": 2,   "levels": { "10": "Flawless…", "5": "Frequent errors…" } },
  "creativity": { "label": "Creativity", "max_score": 5,  "weight": 1,   "levels": {} }
}
```

- **Key:** lowercase letters, numbers and underscores. The AI returns scores under this key.
- **`max_score`:** from 2 to 10. Scores run from 1 to `max_score`.
- **`weight`:** relative importance in the weighted score (0 or more).
- **`levels`:** optional descriptors. A level without a descriptor uses the nearest described level below it.
- **Legacy rubrics** (`{ "grammar": { "4": "…", "3": "…" } }`) still load. Their maximum is the highest level, and built-in criteria keep their default weights.
- **Which rubric scores an essay:**
  1. the rubric the teacher picked on the Evaluate page (their own or the school-wide one);
  2. otherwise the teacher's default rubric;
  3. otherwise the active school-wide rubric;
  4. otherwise the built-in rubric: Main Statement, Organization, Evidence, Analysis and Grammar, each 1–4, with Evidence and Analysis weighted 1.5×.

  There is one active rubric per scope: one school-wide, and one default per teacher. A teacher can't score with another teacher's private rubric.

**Score math** (shared by backend and frontend):
- Normalized score of one criterion: `(score − 1) / (max − 1)`, from 0 to 1.
- Weighted score: weight-averaged normalized scores, shown as a percentage.
- Grade bands on the weighted score: Excellent ≥ 83%, Good ≥ 50%, Developing ≥ 17%, otherwise Beginning.

**Limitation:** `knowledge_base` only has columns for the five built-in criteria, so scores for admin-added criteria aren't stored when an evaluation is saved to the Knowledge Base.

---

## Prompt Playground statistics

Three slots (A, B, C) send the same prompt to the chosen models in parallel. Each slot can use DeepSeek, Gemma 2 2B, Llama 3.2 3B or Phi-3.5. The prompts are built automatically from the rubric that applies to the user, so nobody edits prompts by hand. Admins can still open **Advanced (admin): Edit Prompts** to change them. Once an admin edits them, rubric changes no longer overwrite their version until they click **Rebuild from rubric**.

**Scoring difference table:** each model's score per criterion, the consensus (mean of the three), the spread (max − min), every pairwise difference, and which model was closest per criterion. Rows where models are more than one rubric level apart are highlighted.

**Reference:** what the models are compared against.
- *Consensus*: mean of the three models.
- *System evaluator*: this app's own `/evaluate` result.
- *Teacher scores*: scores you type in, to find which model grades most like you.

All distances use the normalized 0–100 scale, so criteria with different ranges are comparable.

| Statistic | Meaning |
|---|---|
| Similarity | 100 − average distance from the reference |
| Avg distance | Mean absolute difference from the reference, in normalized points |
| Bias / tendency | Mean signed difference: above +8 = **Lenient**, below −8 = **Strict**, otherwise **Balanced** |
| Criteria closest | Number of criteria where this model was nearest the reference |
| Differentiation | Standard deviation of the model's own scores. Low means it gives every criterion about the same score. |
| Exact match / within ±1 | Pairwise agreement between two models, per criterion |
| Correlation r | Pearson correlation of two models' normalized scores (needs 3+ criteria with variation) |

**Save this run** adds the result to a per-browser history. The history then shows, across runs, how often each model was closest to your chosen reference, along with its average similarity and tendency.

---

## OCR and handwriting

- **Images:** the original and a preprocessed copy are both run through OCR. TrOCR output is preferred for handwriting, otherwise the longest result wins. The text is then corrected by DeepSeek (with Ollama as fallback) under strict limits: the length ratio must stay within 0.85–1.15 and similarity must be at least 0.85, so the AI can only fix misread characters, not rewrite the essay.
- **English only:** Tesseract runs with the English model only (`-l eng`), and TrOCR is an English handwriting model.
- **PDF:** processed as a background job, with the frontend checking `/ocr/status/{job_id}` every 1.5 s. Each page is handled on its own:
  - pages with a text layer are read directly (instant);
  - blank pages are skipped;
  - scanned or handwritten pages are OCR'd in parallel (`PDF_OCR_WORKERS`). TrOCR pages share one model and take turns, while Tesseract pages run alongside them.

  Tesseract reads each page once, returning text and confidence together. TrOCR uses greedy decoding, batches of 8 lines and at most 64 tokens per line.
- **DOCX:** text is extracted directly, without OCR.
- **Fine-tuning:** admins correct samples in **OCR Labeler**. `backend/export_training_data.py` exports them to `training-data/`, and `backend/train_trocr.py` fine-tunes `microsoft/trocr-base-handwritten` on Google Colab.

---

## API reference

All routes except `/health`, `/ocr*`, `/test-prompt`, `/rubric*`, `/suggestions` and the public `GET /ratings*` require `Authorization: Bearer <Supabase access token>`. Routes marked admin return 403 for other users.

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| POST | `/evaluate`, `/evaluate-rag` | Evaluate `{ text, evaluation_type: "analytic" \| "holistic", rubric_id? }`; `scores.rubric` names the rubric used |
| GET | `/rubric/active?rubric_id=` | The rubric an evaluation would use for the caller (or the given one): info and criteria |
| GET | `/rubric`, `/suggestions` | Built-in rubric and usage guide |
| GET/POST/PUT/DELETE | `/rubrics`, `/rubrics/{id}` | Rubric CRUD: teachers manage their own; `is_system: true` (school-wide) is admin-only |
| POST | `/override` | Teacher override `{ original_essay, original_scores, teacher_scores?, teacher_feedback, suggested_changes?, satisfaction? }` |
| GET | `/learning-kb` | Your overrides |
| POST/GET | `/knowledge`, `/knowledge/{id}` | Knowledge Base |
| POST | `/test-prompt` | `{ ai_provider: deepseek \| gemma \| ollama \| llamacpp, system_prompt, user_prompt, model? }` |
| POST/GET/DELETE | `/saved-essays` | Playground drafts |
| POST | `/ocr` | Upload `.png .jpg .jpeg .bmp .tiff .pdf .docx` |
| GET | `/ocr/status/{job_id}` | PDF job status |
| GET/POST | `/ocr/pending`, `/ocr/correct` | OCR labeling (admin) |
| … | `/surveys…`, `/questions/{id}` | Surveys (admin writes) and responses |
| … | `/ratings…`, `/comments…`, `/reactions` | Rate Us |

Interactive docs: `https://api.essay-evaluator.duckdns.org/docs`.
