# Interview Coach Web

A web application for preparing for technical interviews using your own Markdown notes.

The project works in two modes:
- `Smart` mode: the LLM asks/selects questions, evaluates answers, gives feedback, and periodically analyzes your progress.
- `Simple` mode: a sequential/weighted run through pre-prepared Q/A pairs from your notes, with optional chat discussion.

---

## Table of contents

- [1. What the project does](#about)
- [2. Architecture and data flow](#architecture)
- [3. Repository structure](#repo-structure)
- [4. Requirements and dependencies](#requirements)
- [5. Installation and launch](#setup)
- [6. Configuration via `.env`](#env)
- [7. Notes format (critical)](#content-format)
- [8. How to use the application](#how-to-use)
- [9. WebSocket/HTTP protocol](#protocol)
- [10. Logs, artifacts, and where to look](#logs)
- [11. Development and modifying the code](#dev)
- [12. Common problems and solutions](#troubleshooting)
- [13. Limitations of the current implementation](#limitations)
- [14. Quick checklist before an interview](#checklist)

---

<a id="about"></a>
## 1. What the project does

The core idea: you practice answering questions based on your own materials (`conspects/*.md`), and the system:
- selects a topic and a question;
- shows context from the notes;
- accepts your answer (as text or by voice);
- evaluates the answer via an LLM against a rubric;
- saves the history to `runs/<timestamp>/history.jsonl`;
- periodically builds a summary of your weak/strong topics.

Key features:
- a single web interface (`FastAPI + static HTML/CSS/JS`);
- voice input (`MediaRecorder` in the browser + `faster-whisper` on the server);
- selection of source notes before starting;
- in `Simple` mode, support for source weights (summing to 100%);
- a debug drawer with the events and metadata of the current session.

---

<a id="architecture"></a>
## 2. Architecture and data flow

### 2.1 Components

- `app/main.py`: FastAPI entrypoint, HTTP + WebSocket, initialization of indexes/models.
- `app/coach_engine.py`: `Smart` mode sessions, question selection, evaluation, progress analysis.
- `interview_coach_core/parsing.py`: Markdown parser and building of content indexes.
- `interview_coach_core/openrouter.py`: creation of OpenRouter LLM clients.
- `interview_coach_core/stt.py`: loading and use of `faster-whisper`.
- `app/static/*`: client side (UI, events, WebSocket client, voice UX).

### 2.2 Flow in Smart mode

1. The client connects to `/ws`.
2. The server sends `hello` (flags `has_llm`, `has_stt`, the list of sources, the config).
3. The client sends `start` with `mode: "smart"` and the selected sources.
4. `CoachEngine.start_or_next_question` selects a topic and a question, generating a question/rubric if needed.
5. The server sends `question`.
6. The client sends `answer`.
7. `CoachEngine.submit_answer` calls the Evaluator LLM with `EVAL_SYSTEM` and validates the JSON via pydantic.
8. The server sends `feedback`, updates the counter, and writes to `history.jsonl`.
9. Every `ANALYZE_EVERY_N` answers, `run_analysis` is launched and an `analysis` message arrives.

### 2.3 Flow in Simple mode

1. The client sends `start` with `mode: "simple"`, the sources, and the weights.
2. The server builds `SimpleSessionState` (per-file question queues, normalized weights).
3. On `next`, the next source is chosen by weight and a `simple_question` is returned.
4. On `simple_skip`, the current question goes into the repeat queue.
5. When the main questions run out, questions from `skipped_queue` are served.
6. After the run is fully complete, the server sends `simple_done`.
7. `simple_chat` messages go to the LLM (if an OpenRouter key is configured).

---

<a id="repo-structure"></a>
## 3. Repository structure

```text
.
├── app/
│   ├── main.py
│   ├── coach_engine.py
│   ├── __init__.py
│   └── static/
│       ├── index.html
│       ├── app.js
│       └── style.css
├── interview_coach_core/
│   ├── config.py
│   ├── parsing.py
│   ├── strategy.py
│   ├── prompts.py
│   ├── models.py
│   ├── logging_utils.py
│   ├── openrouter.py
│   ├── stt.py
│   └── __init__.py
├── conspects/
│   └── *.md
├── runs/
├── .env
├── .gitignore
└── requirements.txt
```

### 3.1 What is responsible for what (files)

| Path | Purpose |
|---|---|
| `app/main.py` | Application initialization, startup loading of indexes/models, the `/`, `/ws`, `/stt` endpoints |
| `app/coach_engine.py` | Smart session state, question selection, calling the evaluator/analyst, writing history |
| `app/static/app.js` | UI logic, handling ws messages, Smart/Simple modes, voice input |
| `app/static/index.html` | Interface markup (topbar, question card, feed, drawers) |
| `app/static/style.css` | Visual styles and responsive behavior |
| `interview_coach_core/config.py` | Dataclass config from env variables |
| `interview_coach_core/parsing.py` | Parsing markdown, extracting questions, building indexes |
| `interview_coach_core/strategy.py` | Heuristics for selecting topics and questions |
| `interview_coach_core/prompts.py` | System prompts and a robust JSON parser |
| `interview_coach_core/models.py` | Pydantic schemas for LLM responses |
| `interview_coach_core/logging_utils.py` | Working with `runs/`, JSONL logging, progress aggregation |
| `interview_coach_core/openrouter.py` | Configuring the `ChatOpenAI` client for OpenRouter |
| `interview_coach_core/stt.py` | Loading the whisper model and transcribing audio |
| `conspects/*.md` | Sources of questions/theory for both modes |
| `runs/` | Session artifacts: history, topics snapshot |

---

<a id="requirements"></a>
## 4. Requirements and dependencies

Minimum:
- Python 3.9+ (3.10/3.11 recommended);
- internet access for OpenRouter calls (for `Smart` mode and `Simple chat`);
- a microphone + a browser with `MediaRecorder` (for voice input);
- `ffmpeg` in the system for stable audio processing with `faster-whisper`.

Python dependencies (`requirements.txt`):
- `fastapi>=0.110`
- `uvicorn[standard]>=0.23`
- `langchain>=0.2`
- `langchain-openai>=0.1.20`
- `pydantic>=2.0`
- `python-dotenv>=1.0`
- `faster-whisper>=1.0`
- `soundfile>=0.12`

---

<a id="setup"></a>
## 5. Installation and launch

### 5.1 Quick start

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open in your browser:
- `http://127.0.0.1:8000`

### 5.2 Important note about content

On startup, the application always reads `NOTES_DIR` (default `conspects/`).
If there are no `*.md` files in the folder, startup will fail with an error.

### 5.3 If you only need Simple without evaluation

`Simple` mode works without `OPENROUTER_API_KEY` for showing Q/A.
But chat in `Simple` mode (`simple_chat`) will be unavailable without a key.

---

<a id="env"></a>
## 6. Configuration via `.env`

`Config.from_env()` reads the following variables:

| Variable | Default | What it does |
|---|---|---|
| `NOTES_DIR` | `conspects` | Folder with the Markdown notes |
| `RUNS_DIR` | `runs` | Folder for session artifacts |
| `ANALYZE_EVERY_N` | `30` | Frequency of progress analysis in Smart mode |
| `MAX_QUESTIONS_PER_SOURCE` | `0` | Question limit per file (`0` = no limit) |
| `OPENROUTER_API_KEY` | `""` | OpenRouter key |
| `OPENROUTER_MODEL` | `deepseek/deepseek-chat` | Model for generation/eval |
| `APP_HTTP_REFERER` | `http://localhost` | The `HTTP-Referer` header for OpenRouter |
| `APP_TITLE` | `InterviewCoachWeb` | The `X-Title` header for OpenRouter |
| `LLM_TIMEOUT_S` | `120` | LLM request timeout (sec) |
| `WHISPER_MODEL` | `large-v3` | The `faster-whisper` model |
| `WHISPER_DEVICE` | `cpu` | `cpu` or `cuda` |
| `WHISPER_COMPUTE_TYPE` | `int8` | Compute type (`int8`, `float16`, ...) |
| `WHISPER_INITIAL_PROMPT` | (tech prompt) | A hint for terms during STT |
| `WHISPER_LANGUAGE` | `""` | Language for STT (`""` = auto-detect) |
| `WHISPER_BEAM_SIZE` | `2` | Beam size (smaller = faster) |
| `WHISPER_CONDITION_ON_PREV` | `true` | Take the previous text into account during decoding |

Example `.env`:

```env
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_MODEL=deepseek/deepseek-chat
NOTES_DIR=conspects
RUNS_DIR=runs
ANALYZE_EVERY_N=30
MAX_QUESTIONS_PER_SOURCE=0
WHISPER_MODEL=large-v3
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
```

---

<a id="content-format"></a>
## 7. Notes format (critical)

The project uses two different ways of extracting content.

### 7.1 Format for Smart mode (`build_content_index`)

`Smart` topics are built from `###` or `####` sections.

Rules:
- only heading levels 3 and 4 are turned into topics;
- the section heading = `topic.title`;
- the section text = `topic.theory`;
- questions inside a section are extracted by the `extract_questions` heuristics.

`extract_questions` recognizes:
- lines with the prefixes `Q:`, `Q -`, `Вопрос:`, `Question:`;
- list/numbered items if they contain `?` or start with `how/why/what/...`;
- short lines up to 200 characters containing `?`.

### 7.2 Format for Simple mode (`build_simple_index`)

In `Simple` mode, question-answer pairs are required:
- the question heading: `### Вопрос: ...` or `#### Вопрос: ...`;
- the answer is expected after a line of the form `**Ответ:**`.

Minimal template:

```markdown
## Вопросы

### Тема

#### Вопрос: Что такое overfitting?
**Ответ:**
A short and structured answer...
```

For `Simple` mode, the `subtopic` from the nearest `###` heading is also stored.

### 7.3 Practical content recommendations

- keep one logical domain per file (`Python`, `NLP`, `SQL`, etc.);
- write the question as specifically as possible;
- in the answer, state the criteria for a good answer, not just a definition;
- avoid huge sections without `###/####` headings, otherwise no topics will be extracted.

---

<a id="how-to-use"></a>
## 8. How to use the application

### 8.1 Interface

- top bar: status, the `Smart/Simple` toggle, the `Start/Next`, `New session`, `Debug` buttons;
- question card: the current question, source, progress, a toggle to show the notes/answer;
- feed: the user's answers, feedback, analysis, chat;
- the `Topic selection` drawer: file selection, and in `Simple` mode additionally weight configuration.

### 8.2 Smart mode scenario

1. Make sure `OPENROUTER_API_KEY` is set.
2. Open the source selection drawer and check the notes you need (optional).
3. Click `Start`.
4. Answer by text or via the microphone.
5. View the `feedback` after each answer.
6. Every `ANALYZE_EVERY_N` questions, an `analysis` block arrives.

### 8.3 Simple mode scenario

1. Switch the toggle to `Simple`.
2. Select the notes and set the weights (auto-normalized to 100%).
3. Click `Start`.
4. Click `Next` for the next question.
5. If you want to repeat a question later, click `Skip`.
6. Use the input field as a chat about the current question/reference answer.

### 8.4 Voice input

How it works:
- the `🎙️` button starts recording;
- pressing it again stops recording;
- the audio is sent to `POST /stt`;
- the transcription is inserted into the input field;
- then you send it with the usual `Send` button.

---

<a id="protocol"></a>
## 9. WebSocket/HTTP protocol

### 9.1 HTTP endpoints

- `GET /` -> UI (`index.html`)
- `POST /stt` -> `{ text, language, duration }`
- `GET /static/*` -> client assets

### 9.2 WebSocket endpoint

- `WS /ws`

#### Client -> server

- `start`: launch a mode (`smart` or `simple`)
- `answer`: the user's answer in `smart`
- `simple_chat`: a chat message in `simple`
- `next`: the next question in `simple`
- `simple_skip`: skip the current question in `simple`
- `reset`: reset the session, receive a new `hello`

#### Server -> client

- `hello`: session capabilities and config
- `ready`: the `run_dir` and `history_path` paths
- `status`: the agent/stage status
- `event`: service events
- `question`: a Smart mode question
- `simple_question`: a Simple mode question
- `simple_done`: completion of a Simple run
- `message`: system or user messages
- `chat`: Simple mode chat messages
- `feedback`: evaluation of an answer in Smart mode
- `analysis`: periodic progress analysis
- `counter`: the value of the answer counter
- `need_input`: a signal that input is awaited

---

<a id="logs"></a>
## 10. Logs, artifacts, and where to look

A directory is created for each Smart session:
- `runs/YYYYMMDD_HHMMSS/`

Inside:
- `history.jsonl`: the full log of questions/answers/evaluations;
- `topics_snapshot.json`: a snapshot of the available topics at the moment of start.

The record format in `history.jsonl` includes:
- `ts`, `n`, `topic_id`, `topic_title`;
- `question`, `user_answer`, `expected`;
- `score`, `short_verdict`;
- `missing_points`, `incorrect_points`, `improvement_tips`, `ideal_answer`;
- `model`.

For quick diagnostics:
- the `Debug` UI drawer shows `event` messages;
- the `run_dir`/`history_path` paths are also visible there.

---

<a id="dev"></a>
## 11. Development and modifying the code

### 11.1 Local development

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Changes:
- backend (`app/*.py`, `interview_coach_core/*.py`) -> reloaded via `--reload`;
- frontend (`app/static/*`) -> a page refresh is enough.

### 11.2 Where to make changes for various tasks

- new topic/question selection logic -> `interview_coach_core/strategy.py`;
- changes to the evaluation/analysis prompts -> `interview_coach_core/prompts.py`;
- a new markdown parsing format -> `interview_coach_core/parsing.py`;
- changes to the ws protocol -> `app/main.py` and `app/static/app.js` in sync;
- UI/UX -> `app/static/index.html`, `app/static/style.css`, `app/static/app.js`.

### 11.3 Tests

There is currently no separate test suite in the repository.
If you add functionality, the following minimum is recommended:
- unit tests for `parsing.py`;
- a smoke test of the ws scenario `start -> question -> answer`.

---

<a id="troubleshooting"></a>
## 12. Common problems and solutions

### 12.1 `OPENROUTER_API_KEY is not set`

Symptom:
- in Smart mode, the system reports a configuration error.

What to do:
- add `OPENROUTER_API_KEY` to `.env`;
- restart `uvicorn`.

### 12.2 Whisper does not load / `/stt` returns 503

Symptom:
- `has_stt=false` in `hello`;
- `/stt` responds `Whisper model not loaded`.

What to do:
- check `WHISPER_MODEL`, `WHISPER_DEVICE`, `WHISPER_COMPUTE_TYPE`;
- make sure there are enough resources for the chosen model;
- install/verify `ffmpeg`.

### 12.3 An error about missing markdown on startup

Symptom:
- startup fails with the error `No .md files found in ...`.

What to do:
- put `.md` files in the `NOTES_DIR` directory (default `conspects/`).

### 12.4 JSON from the LLM sometimes breaks

Mitigation already exists in the code:
- `robust_json_load` tries to extract the JSON block;
- on error, a recovery prompt `Return ONLY valid JSON...` is launched.

What else can be done:
- lower the temperature/switch the model;
- tighten the schema wording in `prompts.py`.

### 12.5 No questions appear in Simple mode

Check:
- whether the files contain the `###/#### Вопрос: ...` + `**Ответ:**` template;
- whether sources are checked in the drawer;
- whether `simple_index` in `hello` is non-empty.

---

<a id="limitations"></a>
## 13. Limitations of the current implementation

- no auth/accounts and no server-side multi-user model;
- no database, everything is stored in the file system;
- the content parser is based on heuristics and is sensitive to the Markdown structure;
- no dedicated test package;
- no rate-limit/retry/backoff layer for the LLM API.

---

<a id="checklist"></a>
## 14. Quick checklist before an interview

1. Check `.env` (`OPENROUTER_API_KEY`, `OPENROUTER_MODEL`).
2. Check the structure of `conspects/*.md` for both modes.
3. Run 30+ questions in Smart mode (at least 1 analysis cycle).
4. Based on the `analysis`, select your weak topics and go through them in Simple mode with an increased weight.
5. Review `runs/*/history.jsonl` and write down recurring mistakes.

---

If you need a separate README for deployment (Docker/systemd/reverse proxy), it is better to move it to `docs/DEPLOY.md` so that this file stays focused on local development and interview preparation.
