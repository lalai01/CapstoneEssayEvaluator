# AI Essay Evaluator

A web app that scores essays against a configurable rubric, gives written feedback, reads handwritten and printed essays with OCR, and lets teachers correct the AI so later evaluations improve.

- **Frontend:** React 19 + Vite + Tailwind, hosted on Firebase Hosting (`https://capstoneessayevaluator.web.app`)
- **Backend:** FastAPI (Python 3.11) in Docker on a Contabo VPS (`https://essay-evaluator.duckdns.org`, set as `VITE_API_URL`)
- **Data and auth:** Supabase (Postgres, Auth with Google/Yahoo, Storage)
- **AI models:** DeepSeek API and Ollama (Llama 3.2 3B, Qwen 2.5 1.5B), reachable from the backend. llama.cpp (Phi-3.5 Mini) is still supported by the API but no longer offered in the Playground, because it was too slow on the VPS CPU.

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
| **Evaluate Essay** | Paste text or upload an image, PDF or DOCX, then choose a scoring mode. **Analytic** gives a separate score for each rubric criterion (e.g. thesis, organization, evidence, grammar; teachers can add others such as vocabulary or transitions) with detailed per-criterion feedback; it takes longer. **Holistic** gives one overall 1–5 score for coherence, organization, style and content together, from a single short AI call with no second feedback call, so it's faster and cheaper; if that call fails, the score is derived from the analytic criteria and the result says so. Optional RAG adds feedback from similar past essays. **Three AIs** (DeepSeek, Llama 3.2 and Gemma 2) evaluate every essay at once; their results appear side by side as each finishes, and the user picks the one they prefer; see [Three AIs per essay](#three-ais-per-essay). |
| **Highlighted essay** | After an evaluation, the essay is shown with the parts to fix highlighted: grammar/spelling errors (red, from the AI's grammar fixes), passages the recommendations refer to (violet, the AI quotes them), vague wording such as "very" or "a lot" (amber), and sentences over 30 words (blue underline). Click a highlight to see the suggested change; each type can be switched off. Works for both scoring modes. |
| **Evaluation report** | **Generate Evaluation Report** creates a multi-page PDF. It includes student, class and teacher details, the rubric used, the total, percentage, weighted score and grade, a per-criterion breakdown (with override scores beside the AI's), strengths and focus areas, criterion comments, feedback, grammar fixes, recommendations, and a teacher comment and signature area. The optional appendix shows the **essay with the same highlights as on screen**: colored backgrounds, a blue underline for long sentences, and numbered [n] marks explained in a **Marked Issues** list. Teachers can also add an **AI-revised reference version** (`POST /revise-essay`, teachers only). It keeps the student's position and ideas, fixes the marked issues, never invents facts (it leaves [placeholders] where evidence is needed), and is clearly labelled as AI-generated, not the student's work. |
| **Satisfaction rating** | After an evaluation, the user rates it 1–10, colored red (1–3), amber (4–6) or green (7–10) wherever it appears. The rating is stored with Knowledge Base entries and teacher overrides. |
| **Override** (everyone signed in) | **Override** an evaluation: enter your own score per criterion, **remarks** (required, with the AI feedback shown for reference) and **suggestions** (optional). Overrides go to the Learning Hub. A **teacher's** override is recorded as approved, feeds back into RAG, and fills the teacher comments in the evaluation report; a student's or admin's override is kept for themselves and doesn't change what the AI learns. |
| **Knowledge Base** | Saved evaluations. A teacher's save is **✅ Approved** (and used by RAG); a student's save is kept privately as **not approved**. Each entry shows which AI scored it, its rubric, its approval, its satisfaction, and whether the essay was also overridden. Search, sort, and filter by AI model, approval, and override status. |
| **Learning Hub** | Your overrides, with essay title, the AI that was overridden, AI-vs-teacher scores, remarks and suggestions. The same search, sort and AI filter, plus filters for "scores changed" and "also in Knowledge Base". |
| **AI Prompt Playground** | Runs the same prompt on three models side by side. A **Score Comparison** table (Category / Primary / Compare / Diff) compares any two of them and can be saved as a PDF report. Agreement statistics are described [below](#prompt-playground-statistics). |
| **Rubrics** (everyone signed in) | Users create their own rubrics. Each criterion has its own maximum score (1–2 up to 1–10), weight and level descriptors, and users can add criteria of their own. A user's **default** rubric is used for their evaluations. They can copy the school-wide rubric as a starting point. Personal rubrics are private to their owner. **Import from PDF/Excel:** upload or drag a rubric file (PDF, .xlsx or .csv) and its criteria fill the editor (`POST /rubrics/import`, nothing is saved until you save). Gemini reads the PDF itself, so any layout works (criteria as rows or columns, numbered or named levels such as Excellent/Adequate/Needs Work); DeepSeek reads the extracted text if Gemini is unavailable, and a built-in table reader handles simple grids when no AI is. Criteria can then be changed, removed or added; a criterion whose name is already in the rubric is marked **⚠️ Already exists**, and a title you already use is flagged too. |
| **School Rubric** (admin only) | Admins create and activate the school-wide rubric, the default for every teacher without their own. Teachers can view and copy it but not change it; the backend rejects such changes with 403. |
| **Navigation** | A top bar shows the main pages to everyone, plus a **My Work** menu (Knowledge Base, Learning Hub, My Rubrics) for everyone signed in and an **Admin** menu (Admin Dashboard, School Rubric, Manage Surveys, OCR Labeler) for admins. People only see menus their role allows; on small screens the menu collapses into a drawer. The account menu shows the email, role badges and Sign out. |
| **Admin Dashboard** (admin only) | Add teachers and admins by email (no SQL needed). Each teacher is **Active**, **Inactive** (keeps the record but removes teacher rights until reactivated) or **Removed** (a soft delete, restorable). Summary cards show the counts, and an **activity log** records who added, removed, activated or deactivated whom, and when. |
| **Surveys** | Admins create surveys; logged-in users answer them in the **Surveys** tab. |
| **OCR Labeler** (admin) | Verify the transcripts of uploaded handwritten pages (prefilled line by line with what the reader saw) or discard unusable photos, to build a handwriting dataset. |
| **Rate Us** | Public ratings with threaded comments and reactions. |

**Roles.** Every signed-in user has one role, decided only by the backend (`GET /me`) and shown as a tag next to their email:

| Role | Can |
|---|---|
| **Student** (default) | Evaluate essays, save them privately (not approved), override the AI for their own records, create their own rubrics, use the Playground |
| **Teacher · Approver** | Everything a student can, plus approve evaluations into the Knowledge Base; their overrides teach the AI |
| **Admin** | Run the system: manage roles, surveys, OCR labels and the **school-wide rubric**. Admins are **not** teachers: their saves and overrides don't teach the AI unless their email is also added as a teacher (then they hold both roles) |

These rules are enforced on the server: a student who calls the API directly still can't override or approve. Teachers are managed in the **🛡️ Admin Dashboard** (stored in the `teachers` table); only **active**, not-removed teachers have teacher rights. A user whose `app_metadata.role` is `teacher` is also a teacher. Admin and teacher are separate roles; one person can hold both.

**Who is an admin** is decided only by the backend (`GET /me`). No admin emails are stored in this repository. A user is an admin if any of these is true:
1. their email is in the `admin_users` table in Supabase. Add the first admin with SQL (see [Database](#database-supabase)), then manage the rest in the **👥 Admins** tab;
2. their Supabase `app_metadata.role` is `admin`;
3. their email is in the optional `ADMIN_EMAILS` environment variable on the server (empty by default).

Email-based admin access only counts once the email is **confirmed**. `user_metadata` is never trusted, because users can edit it themselves from the browser.

An admin can sign in with Google, Yahoo, email and password (**Sign In → Admin Login**), or **Email Me a Sign-in Link** (passwordless). An email can be made admin before the person has ever signed in.

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
   ├── ocr_utils.py, handwriting_ocr.py, ocr_jobs.py, essay_image_check.py
   │                         Tesseract for print, TrOCR for handwriting, background PDF jobs
   └── rag.py                TF-IDF retrieval over the knowledge base and teacher overrides
   │
   ├──► Supabase (tables, storage bucket "ocr-samples")
   ├──► Ollama   http://ollama:11434       (llama3.2:3b scoring/feedback, qwen2.5:1.5b)
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
  auth.py                 verifies sign-in tokens with Supabase Auth (cached 2 min)
  supabase_client.py      anon + service-role Supabase clients
  rag.py                  similar-essay retrieval
  ocr_utils.py            engine selection, DOCX/PDF extraction
  essay_image_check.py    rejects images that aren't pages of writing
  text_quality.py         checks OCR output is English paragraphs
  gemini_ocr.py           reads handwritten pages with Gemini vision
  handwriting_ocr.py      TrOCR line segmentation and transcription
  ocr_jobs.py             background PDF OCR jobs
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

Requirements: Python 3.11, Tesseract OCR, Java (for LanguageTool), and an Ollama instance with `llama3.2:3b` and `qwen2.5:1.5b` pulled.

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
| `SUPABASE_SERVICE_KEY` | yes | Service-role key: admin management (`admin_users`), admin writes (rubrics, surveys), reading the active rubric past RLS |
| `SUPABASE_JWT_SECRET` | no | No longer used. Tokens are verified by Supabase Auth itself (`/auth/v1/user`), which works with both legacy and new signing keys. |
| `DEEPSEEK_API_KEY` | for DeepSeek | OCR correction and the Playground DeepSeek slot |
| `OLLAMA_URL` | no | Default `http://ollama:11434` |
| `SECOND_LOCAL_MODEL` | no | A second local AI (no longer shown on the Evaluate page; Groq replaced it). Default `qwen2.5:1.5b` (replaced Gemma 2 2B: faster on CPU, more reliable JSON). Pull it in Ollama first. |
| `GROQ_API_KEY` | for the Groq AI | Free key from console.groq.com. The third AI on the Evaluate page is Llama 3.3 70B on Groq (answers in seconds). Without it, or when Groq is over its free-tier limit, Gemini answers instead and the card says so. Server only, never in the frontend or git. |
| `GROQ_MODEL` | no | Default `llama-3.3-70b-versatile` (`llama-3.1-8b-instant` is faster, with higher free limits). |
| `LLAMACPP_URL` | no | Default `http://llamacpp:8080/v1` |
| `LLAMACPP_MODEL` | no | Default `Phi-3.5-mini-instruct-Q4_K_M.gguf` |
| `ADMIN_EMAILS` | no | Optional extra admin emails, comma-separated. Prefer the `admin_users` table so emails stay out of config files. |
| `GUEST_EVALS_PER_HOUR` | no | Evaluations a signed-out visitor may run per hour (per IP). Default `5`; `0` requires sign-in |
| `GUEST_UPLOADS_PER_HOUR` | no | File uploads (OCR) a signed-out visitor may run per hour. Default `10`; `0` requires sign-in |
| `PDF_OCR_WORKERS` | no | Scanned PDF pages OCR'd at once. Default `min(4, CPU cores)` |
| `GEMINI_API_KEY` | no | Google AI Studio key (https://aistudio.google.com/apikey). When set, Gemini vision reads handwritten pages; otherwise the local TrOCR model does |
| `GEMINI_OCR_MODELS` | no | Gemini models to try in order when one is unavailable or out of quota. Default `gemini-3.8-flash,gemini-3.5-flash-lite` |
| `HANDWRITING_BEAMS` | no | Beam width for handwriting OCR. Default `3`; `1` is faster but reads less accurately |
| `TROCR_PATH` | no | Folder with the TrOCR model. Default `/opt/trocr` |
| `TROCR_QUANTIZE` | no | `1` runs TrOCR with int8 quantization: about 2× faster handwriting OCR, slightly less accurate. Off by default. |

### Frontend (`frontend/.env.production`, `frontend/.env.local`)

| Variable | Purpose |
|---|---|
| `VITE_API_URL` | Backend base URL |
| `VITE_SUPABASE_URL` | Supabase project URL |
| `VITE_SUPABASE_ANON_KEY` | Supabase anon key (safe to expose; protect data with RLS) |

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

-- Teachers (approvers), managed in the Admin Dashboard (required for teachers).
create table if not exists teachers (
  email      text primary key,
  added_by   text,
  created_at timestamptz not null default now()
);
alter table teachers enable row level security;

-- Teacher status and soft delete, plus the role activity log (required for the Admin Dashboard).
alter table teachers add column if not exists status     text not null default 'active';
alter table teachers add column if not exists updated_by text;
alter table teachers add column if not exists updated_at timestamptz;
alter table teachers add column if not exists deleted_by text;
alter table teachers add column if not exists deleted_at timestamptz;
create table if not exists role_audit_log (
  id         bigserial primary key,
  role       text not null,          -- 'teacher' | 'admin'
  email      text not null,          -- whose role changed
  action     text not null,          -- added | removed | restored | activated | deactivated
  actor      text,                   -- the admin who made the change
  details    jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);
alter table role_audit_log enable row level security;

-- Who approved an evaluation or override (recommended).
alter table knowledge_base    add column if not exists approved_by text;
alter table learning_feedback add column if not exists approved_by text;

-- Admins managed in the app's Admins tab (required for that tab).
-- RLS on with no policies: only the backend's service-role key can read or write it.
create table if not exists admin_users (
  email      text primary key,
  added_by   text,
  created_at timestamptz not null default now()
);
alter table admin_users enable row level security;

-- Add the first admin(s); after that, use the Admins tab in the app.
insert into admin_users (email, added_by) values ('you@example.com', 'setup')
on conflict (email) do nothing;
```

If every admin is ever removed, add one back with the `insert` above.

**Supabase Auth settings** (Dashboard → Authentication):
- Keep **Confirm email** enabled, so nobody can register an admin's email address with a password without owning the inbox.
- For email sign-in links, add your site (`https://capstoneessayevaluator.web.app`) under **URL Configuration → Redirect URLs**.

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

## Three AIs per essay

Clicking **Evaluate** sends the essay to **DeepSeek** (cloud), **Llama 3.2 3B** and **Qwen 2.5 1.5B** (both on our server, via Ollama) at the same time (`ai_model` = `deepseek` / `llama` / `gemma` with `strict_model: true`). Each card fills in as its AI finishes, with its overall score and its **own** feedback (the AI that scored also writes the feedback). The first real AI result is shown straight away; the user can switch to any other, and the chosen one is what's displayed, saved, overridden and put in the report. In this mode there is no fallback to another model, so each card is genuinely that AI's work; an AI that isn't available says so (rule-based scores, clearly marked).

The API still accepts `ai_model: "auto"` (and `local` for Llama) without `strict_model`, which picks one model per essay:

| Situation | Model | Why |
|---|---|---|
| Essay longer than about 650 words (4,000 characters) | DeepSeek | The local model only reads the first 4,000 characters; DeepSeek reads up to 16,000 |
| Rubric with more than 6 criteria, or scores above 1–6 | DeepSeek | The larger model follows complex rubrics more reliably |
| Anything else | Llama 3.2 3B (local) | Fast enough, free, and the essay stays on the server |
| `DEEPSEEK_API_KEY` not set | Llama 3.2 3B | Nothing else is available |

If the chosen model fails (outside the three-AI mode), Auto tries the other one, then falls back to rule-based scoring. An explicit **local** choice never sends the essay to the cloud, even as a fallback. The result card shows which model scored the essay and why.

## Prompt Playground statistics

Three slots (A, B, C) send the essay to the chosen models. By default: A = DeepSeek, B = Qwen 2.5 1.5B (Ollama), C = Llama 3.2 3B (Ollama). Any slot can use any of the three.

**Speed on the VPS CPU.** DeepSeek runs in the cloud, so it starts immediately. The local models (Ollama and llama.cpp) run **one at a time**, shown as "queued", because running them together makes each slower and causes timeouts. They also get a **compact prompt**: the same rubric with no long example, one-sentence notes, and at most 3 grammar fixes and 3 recommendations. Answers are requested in JSON mode, capped at 1,024 tokens, and **streamed**, so a slow but working model isn't cut off (overall limit 280 s). Ollama keeps models loaded for 30 minutes (`OLLAMA_KEEP_ALIVE`) to skip reload time. Each slot shows its time and tokens per second.
 The prompts are built automatically from the rubric that applies to the user, so nobody edits prompts by hand. Admins can still open **Advanced (admin): Edit Prompts** to change them. Once an admin edits them, rubric changes no longer overwrite their version until they click **Rebuild from rubric**.

**Scoring difference table:** each model's score per criterion, the consensus (mean of the three), the spread (max − min), every pairwise difference, and which model was closest per criterion. Rows where models are more than one rubric level apart are highlighted.

**Reference:** what the models are compared against.
- *Consensus*: each model is compared with the average of the **other** models, never with itself. With one model finished there is nothing to compare yet; with two, they are compared with each other and always tie.
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

- **Is it an essay?** Before any OCR, `essay_image_check.py` checks that the image looks like a page of writing: enough letter-sized marks, most of them sitting in rows next to other letters, a light mostly-uncoloured background. Photos of faces, cars, food and so on fail this and `/ocr` returns `422 {"code": "not_essay"}`. The frontend shows this as an "attach a proper essay" dialog instead of continuing. The same applies when the text that comes out isn't English paragraphs (`text_quality.py`: at least 8 words, at least half of them real English words), including scanned PDFs and DOCX files.
- **Images:** Tesseract reads the page first. If it returns a confident, fully English printed page, that's used and nothing else runs. Otherwise, when `GEMINI_API_KEY` is set, **Gemini vision** reads the whole page in a few seconds. It is told to copy the text exactly, keeping the student's spelling, grammar and punctuation mistakes (they're graded), so its text skips the AI correction step. Words it can't read come back as `[illegible]` and lower the confidence. Note that with a key set, handwritten essay photos are sent to Google. Without a key, or when Gemini is unavailable or out of quota, TrOCR reads it too, and whichever result has the larger share of real English words wins (handwriting gets a small preference). TrOCR first straightens the page (deskew), finds each line from the gaps between rows of letters, and cuts out each line with only its own strokes so letter tails from neighbouring lines don't confuse the model. The text is then corrected by DeepSeek (with Ollama as fallback) under strict limits: the length ratio must stay within 0.85–1.15 and similarity must be at least 0.85, so the AI can only fix misread characters, not rewrite the essay.
- **English only:** Tesseract runs with the English model only (`-l eng`), and TrOCR is an English handwriting model.
- **PDF:** processed as a background job, with the frontend checking `/ocr/status/{job_id}` every 1.5 s. Each page is handled on its own:
  - pages with a text layer are read directly (instant);
  - blank pages are skipped;
  - scanned or handwritten pages are OCR'd in parallel (`PDF_OCR_WORKERS`). TrOCR pages share one model and take turns, while Tesseract pages run alongside them.

  Tesseract reads each page once, returning text and confidence together. TrOCR uses beam search (`HANDWRITING_BEAMS`, default 3), batches of 8 lines and at most 64 tokens per line.
- **DOCX:** text is extracted directly, without OCR.
- **Short essays:** when an essay has fewer than 150 words, **Evaluate** asks for confirmation first, because short essays give less reliable scores. Under 20 words it can't be evaluated at all, and the dialog says so.
- **Fine-tuning:** every handwritten page that's read is queued in **OCR Labeler** (printed pages aren't), prefilled with the reader's own text before any AI correction, one line per written line (Gemini returns the page's lines as well as the essay). Admins fix it to match the page exactly, or discard it; saving and discarding are admin-only. A **Reader accuracy** panel at the top (`/ocr/accuracy`) measures each reader's character and word error rate on the verified pages (its own text before AI correction vs the verified transcription), so the choice of Gemini over the local model is backed by measurements on our own dataset. Gemini already reads handwriting well, so this dataset mainly measures accuracy and trains the offline fallback reader. `backend/export_training_data.py` cuts them into lines the same way the app reads them and writes `training-data/`; `--pages <folder>` does the same for a folder of page images whose `.txt` transcript has one text line per written line. `backend/train_trocr.py` fine-tunes `microsoft/trocr-base-handwritten` on a GPU (Kaggle; setup at the top of the script) using `training-data/` plus the IAM lines in `handwriting_dataset/` and `full_dataset/`. It keeps the epoch with the lowest character error rate and saves `training_report.png` (loss, CER and accuracy per epoch, CER per handwriting style), `sample_predictions.png` and `training_history.csv` next to the model.

---

## API reference

All routes except `/health`, `/ocr*`, `/test-prompt`, `/rubric*`, `/suggestions` and the public `GET /ratings*` require `Authorization: Bearer <Supabase access token>`. Routes marked admin return 403 for other users. `POST /evaluate`, `/evaluate-rag` and `/ocr` also work for guests (no token): they use the school rubric, can't save anything, and are limited per IP per hour (`GUEST_EVALS_PER_HOUR`, `GUEST_UPLOADS_PER_HOUR`; 429 `{"code": "guest_limit"}` when exceeded). Signed-in users aren't limited.

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| POST | `/evaluate`, `/evaluate-rag` | Evaluate `{ text, evaluation_type: "analytic" \| "holistic", rubric_id? }`; `scores.rubric` names the rubric used |
| GET | `/rubric/active?rubric_id=` | The rubric an evaluation would use for the caller (or the given one): info and criteria |
| GET | `/rubric`, `/suggestions` | Built-in rubric and usage guide |
| GET/POST/PUT/DELETE | `/rubrics`, `/rubrics/{id}` | Rubric CRUD: teachers manage their own; `is_system: true` (school-wide) is admin-only |
| POST | `/override` | Override (any signed-in user; approved only for teachers) `{ original_essay, original_scores, teacher_scores?, teacher_feedback, suggested_changes?, satisfaction? }` |
| GET | `/learning-kb` | Your overrides |
| POST/GET | `/knowledge`, `/knowledge/{id}` | Knowledge Base |
| POST | `/test-prompt` | `{ ai_provider: deepseek \| gemma \| ollama \| llamacpp, system_prompt, user_prompt, model? }` |
| POST/GET/DELETE | `/saved-essays` | Playground drafts |
| POST | `/ocr` | Upload `.png .jpg .jpeg .bmp .tiff .pdf .docx` |
| GET | `/ocr/status/{job_id}` | PDF job status |
| GET/POST | `/ocr/pending`, `/ocr/correct` | OCR labeling (admin) |
| … | `/surveys…`, `/questions/{id}` | Surveys (admin writes) and responses |
| … | `/ratings…`, `/comments…`, `/reactions` | Rate Us |

Interactive docs: `https://essay-evaluator.duckdns.org/docs`.
