const $ = (id) => document.getElementById(id);

const subtitle = $("subtitle");
const pillState = $("pillState");

const qTitle = $("qTitle");
const qMeta = $("qMeta");
const qBody = $("qBody");
const qNotes = $("qNotes");
const qProgress = $("qProgress");
const btnToggleNotes = $("btnToggleNotes");
const sourcePicker = $("sourcePicker");
const sourcesDrawer = $("sourcesDrawer");
const sourcesBackdrop = $("sourcesBackdrop");
const btnCloseSources = $("btnCloseSources");
const sourceTitle = $("sourceTitle");
const weightFooter = $("weightFooter");
const weightTotal = $("weightTotal");
const weightHint = $("weightHint");

const modeToggle = $("modeToggle");
const modeLabelSmart = $("modeLabelSmart");
const modeLabelSimple = $("modeLabelSimple");

const feed = $("feed");
const feedCard = $("feedCard");
const events = $("events");
const meta = $("meta");
const hint = $("hint");

const btnStart = $("btnStart");
const btnSkip = $("btnSkip");
const btnSend = $("btnSend");
const btnClear = $("btnClear");
const btnMic = $("btnMic");
const input = $("input");

const btnDebug = $("btnDebug");
const btnCloseDebug = $("btnCloseDebug");
const drawer = $("drawer");
const drawerBackdrop = $("drawerBackdrop");
const btnNew = $("btnNew");

let ws;
let session = { analyze_every_n: 30 };
let mediaRecorder = null;
let mediaStream = null;
let recording = false;
let transcribing = false;
let sttAbort = null;
let sttEnabled = false;
let agentState = "idle";
let agentLabel = "";
let stepLabel = "";
let interactionEnabled = false;
let analysisLocked = false;
let hasStarted = false;
let mode = "smart";
let availableSourcesSmart = [];
let availableSourcesSimple = [];
let simpleCounts = {};
let selectedSourcesSmart = [];
let selectedSourcesSimple = [];
let simpleWeights = {};

function renderMarkdown(md){
  if (window.marked && window.DOMPurify){
    marked.setOptions({ breaks: true });
    const html = marked.parse(md ?? "");
    return addMathStyling(DOMPurify.sanitize(html));
  }
  return addMathStyling(escapeHtml(md ?? "").replaceAll("\n", "<br/>"));
}

function escapeHtml(s){
  return (s ?? "").replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;");
}

const decodeHtml = (() => {
  const el = document.createElement("textarea");
  return (value) => {
    el.innerHTML = value ?? "";
    return el.value;
  };
})();

function addMathStyling(html){
  if (!html) return "";
  const blockRe = /\$\$(.+?)\$\$/gs;
  const inlineRe = /(^|[^\\])\$(.+?)\$/g;
  const canRender = window.katex && typeof window.katex.renderToString === "function";

  let out = html.replace(blockRe, (_, expr) => {
    if (canRender){
      try {
        const rendered = katex.renderToString(decodeHtml(expr).trim(), {
          displayMode: true,
          throwOnError: false,
        });
        return `<div class="math block katex-block">${rendered}</div>`;
      } catch (err){
        return `<div class="math block">${expr}</div>`;
      }
    }
    return `<div class="math block">${expr}</div>`;
  });

  out = out.replace(inlineRe, (_, lead, expr) => {
    if (canRender){
      try {
        const rendered = katex.renderToString(decodeHtml(expr).trim(), {
          displayMode: false,
          throwOnError: false,
        });
        return `${lead}<span class="math katex-inline">${rendered}</span>`;
      } catch (err){
        return `${lead}<span class="math">${expr}</span>`;
      }
    }
    return `${lead}<span class="math">${expr}</span>`;
  });

  return out;
}

function notesLabel(){
  return mode === "simple" ? "ответ" : "конспект";
}

function updateNotesToggle(expanded){
  const label = notesLabel();
  btnToggleNotes.textContent = expanded ? `Скрыть ${label}` : `Показать ${label}`;
}

function updateStartLabel(){
  if (mode === "simple"){
    btnStart.textContent = hasStarted ? "Далее" : "Начать";
    if (btnSkip){
      const shouldShow = hasStarted && interactionEnabled;
      btnSkip.classList.toggle("hidden", !shouldShow);
      btnSkip.disabled = !interactionEnabled;
    }
  } else {
    btnStart.textContent = "Начать";
    if (btnSkip){
      btnSkip.classList.add("hidden");
      btnSkip.disabled = true;
    }
  }
}

function updateModeUI(){
  document.body.classList.toggle("mode-simple", mode === "simple");
  modeLabelSmart?.classList.toggle("active", mode === "smart");
  modeLabelSimple?.classList.toggle("active", mode === "simple");
  if (modeToggle) modeToggle.checked = mode === "simple";
  input.placeholder = mode === "simple"
    ? "Сообщение…"
    : "Ответ…  (Ctrl+Enter / ⌘+Enter — отправить)";
  updateStartLabel();
  showPrestartCard();
  renderSourcePicker();
  updateInteractionState();
}

function getAvailableSources(){
  return mode === "simple" ? availableSourcesSimple : availableSourcesSmart;
}

function getSelectedSources(){
  return mode === "simple" ? selectedSourcesSimple : selectedSourcesSmart;
}

function normalizeSimpleWeights(){
  if (!selectedSourcesSimple.length) return;
  let total = selectedSourcesSimple.reduce((sum, s) => {
    const val = Number.isFinite(simpleWeights[s]) ? simpleWeights[s] : 0;
    return sum + val;
  }, 0);
  if (total <= 0){
    const base = Math.floor(100 / selectedSourcesSimple.length);
    let remainder = 100 - base * selectedSourcesSimple.length;
    selectedSourcesSimple.forEach((s) => {
      simpleWeights[s] = base + (remainder > 0 ? 1 : 0);
      remainder = Math.max(0, remainder - 1);
    });
    return;
  }
  selectedSourcesSimple.forEach((s) => {
    const val = Number.isFinite(simpleWeights[s]) ? simpleWeights[s] : 0;
    simpleWeights[s] = Math.round(val / total * 100);
  });
  let diff = 100 - selectedSourcesSimple.reduce((sum, s) => sum + (simpleWeights[s] ?? 0), 0);
  let i = 0;
  while (diff !== 0 && selectedSourcesSimple.length){
    const src = selectedSourcesSimple[i % selectedSourcesSimple.length];
    if (diff > 0){
      simpleWeights[src] += 1;
      diff -= 1;
    } else if (simpleWeights[src] > 0){
      simpleWeights[src] -= 1;
      diff += 1;
    }
    i += 1;
  }
}

function applyWeightChange(src, value){
  if (!selectedSourcesSimple.includes(src)) return;
  const parsed = Number(value);
  const v = Number.isFinite(parsed) ? Math.max(0, Math.min(100, Math.round(parsed))) : 0;
  const others = selectedSourcesSimple.filter((s) => s !== src);
  if (!others.length){
    simpleWeights[src] = 100;
    return;
  }
  const remaining = 100 - v;
  let totalOthers = others.reduce((sum, s) => {
    const val = Number.isFinite(simpleWeights[s]) ? simpleWeights[s] : 0;
    return sum + val;
  }, 0);
  if (totalOthers <= 0){
    const base = Math.floor(remaining / others.length);
    let remainder = remaining - base * others.length;
    others.forEach((s) => {
      simpleWeights[s] = base + (remainder > 0 ? 1 : 0);
      remainder = Math.max(0, remainder - 1);
    });
  } else {
    const scaled = {};
    let allocated = 0;
    others.forEach((s) => {
      const base = Number.isFinite(simpleWeights[s]) ? simpleWeights[s] : 0;
      const val = Math.round(base / totalOthers * remaining);
      scaled[s] = val;
      allocated += val;
    });
    let diff = remaining - allocated;
    let i = 0;
    while (diff !== 0 && others.length){
      const s = others[i % others.length];
      if (diff > 0){
        scaled[s] += 1;
        diff -= 1;
      } else if (scaled[s] > 0){
        scaled[s] -= 1;
        diff += 1;
      }
      i += 1;
    }
    others.forEach((s) => { simpleWeights[s] = scaled[s]; });
  }
  simpleWeights[src] = v;
}

function updateWeightTotal(){
  if (!weightTotal) return;
  const total = selectedSourcesSimple.reduce((sum, s) => {
    const val = Number.isFinite(simpleWeights[s]) ? simpleWeights[s] : 0;
    return sum + val;
  }, 0);
  weightTotal.textContent = `${total}%`;
}

function syncWeightUI(){
  if (!sourcePicker) return;
  sourcePicker.querySelectorAll("input.weightRange").forEach((el) => {
    const src = el.dataset.source;
    if (!src) return;
    el.value = simpleWeights[src] ?? 0;
  });
  sourcePicker.querySelectorAll("input.weightNumber").forEach((el) => {
    const src = el.dataset.source;
    if (!src) return;
    el.value = simpleWeights[src] ?? 0;
  });
  updateWeightTotal();
}

function getSimpleWeightsPayload(){
  const payload = {};
  selectedSourcesSimple.forEach((src) => {
    payload[src] = simpleWeights[src] ?? 0;
  });
  return payload;
}

function setStatus(agent, step, state, counter){
  agentLabel = agent || agentLabel || "—";
  stepLabel = step || stepLabel || "";
  if (state) agentState = state;
  if (agent === "ProgressAnalyzer"){
    analysisLocked = state === "running";
    updateInteractionState();
  }
  const labelText = resolveStatusLabel(agentLabel, stepLabel, agentState);
  const dotClass = agentState === "running" ? "good" : agentState === "error" ? "bad" : "";
  pillState.innerHTML = `<span class="pillDot ${dotClass}"></span><span>${labelText}</span>`;
  pillState.title = `Статус: ${labelText}`;

  pillState.style.background =
    agentState === "running" ? "rgba(40,167,69,.14)" :
    agentState === "error" ? "rgba(224,71,85,.16)" :
    "var(--brand-soft)";

  if (typeof counter === "number" && mode === "smart"){
    const every = session.analyze_every_n || 30;
    const until = (every - (counter % every)) || every;
    updateQuestionProgress(counter, every, until);
  }

  if (agentState === "running" && (agent === "TopicPicker" || agent === "QuestionPicker")){
    setQuestionLoading(agent, step);
  }
  if (agentState === "running" && agent === "Evaluator"){
    hint.textContent = "Оцениваю ответ…";
  } else if (agentState === "idle"){
    hint.textContent = "";
  }
}

function setQuestionLoading(agent, step){
  if (!hasStarted) return;
  qTitle.textContent = "Готовлю следующий вопрос…";
  qMeta.textContent = `${agent}: ${step}`;
  qBody.innerHTML = `
    <div class="skeleton">
      <div class="sk sk1"></div>
      <div class="sk sk2"></div>
      <div class="sk sk3"></div>
    </div>
  `;
  qNotes.classList.add("hidden");
  updateNotesToggle(false);
  btnToggleNotes.disabled = true;
  btnToggleNotes.classList.add("ghost");
}

function setQuestion(topicTitle, sourceFile, question, theory){
  qTitle.textContent = topicTitle || "Вопрос";
  qMeta.textContent = sourceFile ? `Источник: ${sourceFile}` : "—";
  qProgress.textContent = qProgress.textContent || "";
  const cleanedQuestion = (question || "").replace(/^["“”'`]+/, "").replace(/["“”'`]+$/, "");
  qBody.innerHTML = `<div class="md">${renderMarkdown(cleanedQuestion)}</div>`;

  const hasNotes = (theory || "").trim().length > 0;
  btnToggleNotes.disabled = !hasNotes;
  btnToggleNotes.classList.add("ghost");
  qNotes.innerHTML = hasNotes ? `<div class="md">${renderMarkdown(theory)}</div>` : "";
  if (!hasNotes){
    qNotes.classList.add("hidden");
    updateNotesToggle(false);
  }
}

function setSimpleQuestion(sourceFile, question, answer, index, total, remaining, subtopic, repeat, repeatRemaining){
  const idx = typeof index === "number" ? index : "";
  qTitle.textContent = idx ? `Вопрос ${idx}` : "Вопрос";
  const metaParts = [];
  if (sourceFile) metaParts.push(`Конспект: ${sourceFile}`);
  const topicLabel = (subtopic || "").trim();
  if (topicLabel) metaParts.push(`Подтема: ${topicLabel}`);
  qMeta.textContent = metaParts.length ? metaParts.join(" · ") : "—";
  if (typeof total === "number" && total > 0){
    const rem = typeof remaining === "number" ? remaining : "";
    const stage = repeat ? "Повтор" : "Основной проход";
    const progressParts = [`${stage}: ${index}/${total}`];
    if (rem !== ""){
      progressParts.push(`Осталось ${rem}`);
    }
    if (repeat && typeof repeatRemaining === "number"){
      progressParts.push(`Повторов в очереди ${repeatRemaining}`);
    }
    qProgress.textContent = progressParts.join(" · ");
  } else {
    qProgress.textContent = "";
  }
  const cleanedQuestion = (question || "").replace(/^["“”'`]+/, "").replace(/["“”'`]+$/, "");
  qBody.innerHTML = `<div class="md">${renderMarkdown(cleanedQuestion)}</div>`;

  const hasAnswer = (answer || "").trim().length > 0;
  btnToggleNotes.disabled = !hasAnswer;
  btnToggleNotes.classList.add("ghost");
  qNotes.innerHTML = hasAnswer ? `<div class="md">${renderMarkdown(answer)}</div>` : "";
  qNotes.classList.add("hidden");
  updateNotesToggle(false);
}

function resolveStatusLabel(agent, step, state){
  if (step && step.trim()) return step;
  const map = {
    TopicPicker: "Выбираю тему",
    QuestionPicker: "Формирую вопрос",
    Evaluator: "Оцениваю ответ",
    ProgressAnalyzer: "Анализирую прогресс",
    Session: "Сессия",
  };
  if (map[agent]) return map[agent];
  if (state === "running") return "В работе";
  if (state === "error") return "Ошибка";
  return "Готово";
}

btnToggleNotes.addEventListener("click", () => {
  if (!hasStarted){
    openSources();
    return;
  }
  if (qNotes.classList.contains("hidden")){
    qNotes.classList.remove("hidden");
    updateNotesToggle(true);
  } else {
    qNotes.classList.add("hidden");
    updateNotesToggle(false);
  }
});

function updateQuestionProgress(counter, every, until){
  if (!qProgress) return;
  const withinCycle = counter === 0 ? 0 : (counter % every || every);
  qProgress.textContent = `Вопрос ${withinCycle}/${every} до анализа`;
  qProgress.title = `До анализа: ${until}`;
}

function updateEmptyState(){
  const hasMessages = feed.querySelector(".msg") !== null;
  const placeholder = $("feedEmpty");
  if (!hasMessages){
    if (!placeholder){
      const div = document.createElement("div");
      div.id = "feedEmpty";
      div.className = "feedEmpty";
      div.textContent = "";
      feed.appendChild(div);
    }
    feedCard?.classList.add("compact");
  } else {
    placeholder?.remove();
    feedCard?.classList.remove("compact");
  }
}

function addUserAnswer(text){
  const div = document.createElement("div");
  div.className = "msg user";
  div.innerHTML = `
    <div class="t"><span>Ответ</span><span style="opacity:.75">вы</span></div>
    <div class="b md">${renderMarkdown(text)}</div>
  `;
  feed.appendChild(div);
  feed.scrollTop = feed.scrollHeight;
  updateEmptyState();
}

function addChatMessage(role, text){
  const div = document.createElement("div");
  const isUser = role === "user";
  div.className = `msg chat ${isUser ? "user" : "assistant"}`;
  div.innerHTML = `
    <div class="t"><span>Чат</span><span style="opacity:.75">${isUser ? "вы" : "LLM"}</span></div>
    <div class="b md">${renderMarkdown(text)}</div>
  `;
  feed.appendChild(div);
  feed.scrollTop = feed.scrollHeight;
  updateEmptyState();
}

function addFeedback(payload){
  const score = payload.score ?? 0;
  const div = document.createElement("div");
  div.className = "msg feedback";
  const bad = score <= 5;

  const missing = (payload.missing_points || []).map(x => `<li>${escapeHtml(x)}</li>`).join("");
  const incorrect = (payload.incorrect_points || []).map(x => `<li>${escapeHtml(x)}</li>`).join("");
  const tips = (payload.improvement_tips || []).map(x => `<li>${escapeHtml(x)}</li>`).join("");

  div.innerHTML = `
    <div class="t">
      <span>Фидбек</span>
      <span class="scorePill ${bad ? "bad" : ""}">${score}/10</span>
    </div>
    <div class="b">
      <div class="md">${renderMarkdown(payload.short_verdict || "")}</div>

      <details>
        <summary>Missing points</summary>
        <div class="inside">${missing ? `<ul>${missing}</ul>` : "—"}</div>
      </details>

      <details>
        <summary>Incorrect / unclear</summary>
        <div class="inside">${incorrect ? `<ul>${incorrect}</ul>` : "—"}</div>
      </details>

      <details>
        <summary>How to improve</summary>
        <div class="inside">${tips ? `<ul>${tips}</ul>` : "—"}</div>
      </details>

      <details>
        <summary>Ideal answer</summary>
        <div class="inside md">${renderMarkdown(payload.ideal_answer || "")}</div>
      </details>
    </div>
  `;
  feed.appendChild(div);
  feed.scrollTop = feed.scrollHeight;
  updateEmptyState();
}

function addAnalysis(title, body){
  const div = document.createElement("div");
  div.className = "msg analysis";
  div.innerHTML = `
    <div class="t"><span>${escapeHtml(title || "Прогресс")}</span><span style="opacity:.75">система</span></div>
    <div class="b md">${renderMarkdown(body || "")}</div>
  `;
  feed.appendChild(div);
  feed.scrollTop = feed.scrollHeight;
  updateEmptyState();
}

function addEvent(agent, level, text){
  const div = document.createElement("div");
  div.className = "ev";
  div.innerHTML = `<div class="h"><span>${escapeHtml(agent)}</span><span>${escapeHtml(level)}</span></div><div class="txt">${escapeHtml(text)}</div>`;
  events.appendChild(div);
  events.scrollTop = events.scrollHeight;
}

function openDrawer(){
  drawer.classList.remove("hidden");
  drawerBackdrop.classList.remove("hidden");
}
function closeDrawer(){
  drawer.classList.add("hidden");
  drawerBackdrop.classList.add("hidden");
}
btnDebug.addEventListener("click", openDrawer);
btnCloseDebug.addEventListener("click", closeDrawer);
drawerBackdrop.addEventListener("click", closeDrawer);

modeToggle?.addEventListener("change", () => {
  const newMode = modeToggle.checked ? "simple" : "smart";
  if (newMode === mode) return;
  mode = newMode;
  resetSession(true);
  updateModeUI();
});

window.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeDrawer();
});

function connect(){
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);

  ws.onopen = () => {
    subtitle.textContent = "Connected";
    setStatus("Session", "Ждет запуска", "idle");
    interactionEnabled = false;
    updateInteractionState();
    updateStartLabel();
    btnStart.disabled = false;
    closeSources();
  };

  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);

    if (msg.type === "hello"){
      session.analyze_every_n = msg.analyze_every_n || 30;
      subtitle.textContent = msg.has_llm ? `Model: ${msg.model}` : "No LLM (set OPENROUTER_API_KEY)";
      hint.textContent = "";
      showPrestartCard();
      availableSourcesSmart = msg.sources || [];
      availableSourcesSimple = msg.simple_sources || [];
      simpleCounts = msg.simple_counts || {};
      selectedSourcesSmart = selectedSourcesSmart.filter((s) => availableSourcesSmart.includes(s));
      selectedSourcesSimple = selectedSourcesSimple.filter((s) => availableSourcesSimple.includes(s));
      if (!selectedSourcesSimple.length && availableSourcesSimple.length){
        selectedSourcesSimple = [...availableSourcesSimple];
      }
      normalizeSimpleWeights();
      renderSourcePicker();
      sttEnabled = Boolean(msg.has_stt);
      btnMic.title = sttEnabled ? "Голосовой ввод" : "Голосовой ввод недоступен";
      interactionEnabled = false;
      updateInteractionState();
      btnStart.disabled = false;
      closeSources();
    }

    if (msg.type === "ready"){
      meta.textContent = `run_dir: ${msg.run_dir}\nhistory: ${msg.history_path}`;
      if (typeof msg.counter === "number"){
        const every = session.analyze_every_n || 30;
        const until = (every - (msg.counter % every)) || every;
        updateQuestionProgress(msg.counter, every, until);
      }
      interactionEnabled = true;
      updateInteractionState();
      btnStart.disabled = true;
    }

    if (msg.type === "status"){
      setStatus(msg.agent, msg.step, msg.state, msg.counter);
    }

    if (msg.type === "counter"){
      setStatus(agentLabel, stepLabel, agentState, msg.counter);
    }

    if (msg.type === "event"){
      addEvent(msg.agent, msg.level, msg.text);
    }

    if (msg.type === "question"){
      setQuestion(msg.topic_title, msg.source_file, msg.question, msg.theory);
      hint.textContent = "";
      interactionEnabled = true;
      updateInteractionState();
      btnStart.disabled = true;
      closeSources();
    }

    if (msg.type === "simple_question"){
      hasStarted = true;
      setSimpleQuestion(
        msg.source_file,
        msg.question,
        msg.answer,
        msg.index,
        msg.total,
        msg.remaining,
        msg.subtopic,
        msg.repeat ?? false,
        msg.repeat_remaining ?? 0
      );
      hint.textContent = "Можно обсудить ответ в чате";
      interactionEnabled = true;
      updateInteractionState();
      btnStart.disabled = false;
      closeSources();
    }

    if (msg.type === "simple_done"){
      hasStarted = true;
      qTitle.textContent = "Вопросы закончились";
      qMeta.textContent = "";
      qProgress.textContent = "";
      qBody.innerHTML = `<div class="md">${renderMarkdown(msg.reason || "Нет доступных вопросов.")}</div>`;
      qNotes.classList.add("hidden");
      updateNotesToggle(false);
      btnToggleNotes.disabled = true;
      interactionEnabled = false;
      updateInteractionState();
      btnStart.disabled = true;
    }

    if (msg.type === "message"){
      if (msg.role === "user"){
        addUserAnswer(msg.body || "");
      } else if (msg.role === "system"){
        addAnalysis(msg.title || "Система", msg.body || "");
      }
    }

    if (msg.type === "chat"){
      addChatMessage(msg.role, msg.text || "");
    }

    if (msg.type === "feedback"){
      addFeedback(msg);
    }

    if (msg.type === "analysis"){
      addAnalysis(msg.title, msg.body);
    }
  };

  ws.onclose = () => {
    subtitle.textContent = "Disconnected (refresh page)";
    interactionEnabled = false;
    updateInteractionState();
    updateStartLabel();
    btnStart.disabled = true;
    closeSources();
  };
}

function autosize(){
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 160) + "px";
}
input.addEventListener("input", autosize);

function sendAnswer(){
  const text = input.value.trim();
  if (!text) return;
  try {
    if (mode === "simple"){
      ws?.send(JSON.stringify({type:"simple_chat", text}));
    } else {
      ws?.send(JSON.stringify({type:"answer", text}));
    }
  } finally {
    input.value = "";
    autosize();
    input.focus();
  }
}
btnSend.addEventListener("click", sendAnswer);
btnClear.addEventListener("click", () => { input.value = ""; autosize(); input.focus(); });

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    sendAnswer();
  }
});

function resetSession(sendReset){
  feed.innerHTML = "";
  events.innerHTML = "";
  showPrestartCard();
  if (sendReset && ws && ws.readyState === WebSocket.OPEN){
    ws.send(JSON.stringify({type:"reset"}));
  }
  updateEmptyState();
  hasStarted = false;
  analysisLocked = false;
  interactionEnabled = false;
  updateInteractionState();
  updateStartLabel();
  btnStart.disabled = false;
  closeSources();
}

btnNew.addEventListener("click", () => {
  resetSession(true);
});

btnStart.addEventListener("click", () => {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  if (mode === "simple"){
    if (!hasStarted){
      hasStarted = true;
      updateStartLabel();
      setStatus("SimpleMode", "Старт…", "running");
      setQuestionLoading("SimpleMode", "Готовлю вопрос…");
      ws.send(JSON.stringify({type:"start", mode:"simple", sources: selectedSourcesSimple, weights: getSimpleWeightsPayload()}));
    } else {
      ws.send(JSON.stringify({type:"next", mode:"simple", weights: getSimpleWeightsPayload()}));
    }
    btnStart.disabled = false;
    closeSources();
    return;
  }
  hasStarted = true;
  setStatus("Session", "Подключение…", "running");
  setQuestionLoading("Server", "Готовлю вопрос…");
  ws.send(JSON.stringify({type:"start", mode:"smart", sources: selectedSourcesSmart}));
  btnStart.disabled = true;
  closeSources();
});

btnSkip?.addEventListener("click", () => {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  if (mode !== "simple" || !hasStarted) return;
  interactionEnabled = false;
  updateInteractionState();
  ws.send(JSON.stringify({type: "simple_skip"}));
});

function updateInteractionState(){
  const enabled = interactionEnabled && (mode === "simple" || !analysisLocked);
  input.disabled = !enabled;
  updateSendClearDisabled();
  updateMicDisabled();
  if (!interactionEnabled){
    btnMic.classList.remove("listening", "loading");
  }
  updateStartLabel();
}

function updateSendClearDisabled(){
  const shouldDisable = !interactionEnabled || recording || transcribing || (mode === "smart" && analysisLocked);
  btnSend.disabled = shouldDisable;
  btnClear.disabled = shouldDisable;
}

function updateMicDisabled(){
  const shouldDisable = !sttEnabled || !interactionEnabled || (mode === "smart" && analysisLocked);
  btnMic.disabled = shouldDisable;
}

function renderSourcePicker(){
  if (!sourcePicker) return;
  sourcePicker.innerHTML = "";
  sourcePicker.classList.add("sourcePicker");
  const sources = getAvailableSources();
  if (!sources.length){
    sourcePicker.classList.add("hidden");
    if (sourceTitle){
      sourceTitle.textContent = mode === "simple"
        ? "Нет вопросов в формате Simple."
        : "Нет доступных конспектов.";
    }
    weightFooter?.classList.add("hidden");
    return;
  }
  sourcePicker.classList.remove("hidden");
  const list = document.createElement("div");
  list.className = "sourceList";

  if (sourceTitle){
    sourceTitle.textContent = mode === "simple"
      ? "Выбери конспекты и веса (всего 100%):"
      : "Выбери конспекты (опционально):";
  }
  if (mode === "simple"){
    weightFooter?.classList.remove("hidden");
  } else {
    weightFooter?.classList.add("hidden");
  }

  sources.forEach((src) => {
    const id = `src_${src.replace(/[^a-zA-Z0-9_-]/g, "_")}`;
    if (mode === "simple"){
      const row = document.createElement("div");
      row.className = "sourceRow";
      const label = document.createElement("label");
      label.className = "sourceItem";
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.id = id;
      cb.value = src;
      cb.checked = selectedSourcesSimple.includes(src);
      cb.addEventListener("change", () => {
        if (cb.checked){
          if (!selectedSourcesSimple.includes(src)) selectedSourcesSimple.push(src);
        } else {
          selectedSourcesSimple = selectedSourcesSimple.filter((s) => s !== src);
          if (!selectedSourcesSimple.length){
            selectedSourcesSimple = [src];
            cb.checked = true;
          }
        }
        normalizeSimpleWeights();
        renderSourcePicker();
      });
      label.appendChild(cb);
      const span = document.createElement("span");
      const count = simpleCounts?.[src];
      span.textContent = count ? `${src} · ${count} вопр.` : src;
      label.appendChild(span);
      row.appendChild(label);

      const weightWrap = document.createElement("div");
      weightWrap.className = "sourceWeight";
      const range = document.createElement("input");
      range.type = "range";
      range.min = "0";
      range.max = "100";
      range.step = "1";
      range.className = "weightRange";
      range.dataset.source = src;
      range.value = simpleWeights[src] ?? 0;
      range.disabled = !selectedSourcesSimple.includes(src);
      range.addEventListener("input", () => {
        applyWeightChange(src, range.value);
        syncWeightUI();
      });
      const num = document.createElement("input");
      num.type = "number";
      num.min = "0";
      num.max = "100";
      num.step = "1";
      num.className = "weightNumber";
      num.dataset.source = src;
      num.value = simpleWeights[src] ?? 0;
      num.disabled = !selectedSourcesSimple.includes(src);
      num.addEventListener("change", () => {
        applyWeightChange(src, num.value);
        syncWeightUI();
      });
      const suffix = document.createElement("span");
      suffix.className = "weightSuffix";
      suffix.textContent = "%";
      weightWrap.appendChild(range);
      weightWrap.appendChild(num);
      weightWrap.appendChild(suffix);
      row.appendChild(weightWrap);
      list.appendChild(row);
    } else {
      const label = document.createElement("label");
      label.className = "sourceItem";
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.id = id;
      cb.value = src;
      cb.checked = selectedSourcesSmart.includes(src);
      cb.addEventListener("change", () => {
        if (cb.checked){
          if (!selectedSourcesSmart.includes(src)) selectedSourcesSmart.push(src);
        } else {
          selectedSourcesSmart = selectedSourcesSmart.filter((s) => s !== src);
        }
      });
      label.appendChild(cb);
      const span = document.createElement("span");
      span.textContent = src;
      label.appendChild(span);
      list.appendChild(label);
    }
  });
  sourcePicker.appendChild(list);
  if (mode === "simple"){
    syncWeightUI();
  }
}

function showPrestartCard(){
  hasStarted = false;
  qTitle.textContent = "Ожидаю запуска";
  qMeta.textContent = "";
  qProgress.textContent = "";
  qBody.innerHTML = "";
  qNotes.classList.add("hidden");
  btnToggleNotes.textContent = mode === "simple" ? "Выбор конспектов" : "Выбор тем";
  btnToggleNotes.disabled = false;
  btnToggleNotes.classList.remove("ghost");
}

function updateMicUI(){
  btnMic.classList.remove("listening", "loading");
  if (recording){
    btnMic.textContent = "⏺️";
    btnMic.classList.add("listening");
    btnMic.title = "Запись... нажми, чтобы остановить";
  } else if (transcribing){
    btnMic.textContent = "⏳";
    btnMic.classList.add("loading");
    btnMic.title = "Расшифровка... нажми, чтобы отменить";
  } else {
    btnMic.textContent = "🎙️";
    btnMic.title = "Голосовой ввод";
  }
  updateSendClearDisabled();
  updateMicDisabled();
}

async function transcribeBlob(blob){
  const fd = new FormData();
  fd.append("file", blob, "audio.webm");
  const started = Date.now();
  transcribing = true;
  updateMicUI();
  sttAbort = new AbortController();
  try{
    const res = await fetch("/stt", { method: "POST", body: fd, signal: sttAbort.signal });
    if (!res.ok){
      addEvent("Voice", "error", `STT HTTP ${res.status}`);
      return;
    }
    const data = await res.json();
    const text = (data.text || "").trim();
    if (text){
      input.value = text;
      autosize();
      input.focus();
      addEvent("Voice", "info", `STT ok (${((Date.now()-started)/1000).toFixed(1)}s)`);
    } else {
      addEvent("Voice", "warn", "Пустая расшифровка");
    }
  } catch (e){
    if (e.name === "AbortError"){
      addEvent("Voice", "warn", "Расшифровка прервана");
    } else {
      addEvent("Voice", "error", e.message || "STT error");
    }
  }
  transcribing = false;
  sttAbort = null;
  updateMicUI();
}

function stopMedia(){
  mediaRecorder = null;
  recording = false;
  if (mediaStream){
    mediaStream.getTracks().forEach(t => t.stop());
    mediaStream = null;
  }
  updateMicUI();
}

async function startRecording(){
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia){
    addEvent("Voice", "error", "Микрофон недоступен");
    return;
  }
  try{
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mime = MediaRecorder.isTypeSupported("audio/webm") ? "audio/webm" : undefined;
    mediaRecorder = new MediaRecorder(mediaStream, mime ? { mimeType: mime } : undefined);
    const chunks = [];
    mediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) chunks.push(e.data);
    };
    mediaRecorder.onstop = async () => {
      const blob = new Blob(chunks, { type: "audio/webm" });
      stopMedia();
      if (blob.size > 0){
        await transcribeBlob(blob);
      }
    };
    mediaRecorder.onerror = (e) => {
      addEvent("Voice", "error", e.error?.message || "Recorder error");
      stopMedia();
    };
    mediaRecorder.start();
    recording = true;
    updateMicUI();
  } catch (e){
    addEvent("Voice", "error", e.message || "Mic error");
    stopMedia();
  }
}

btnMic.addEventListener("click", () => {
  if (recording && mediaRecorder){
    try{ mediaRecorder.stop(); } catch (e){}
    return;
  }
  if (transcribing && sttAbort){
    sttAbort.abort();
    return;
  }
  startRecording();
});

function openSources(){
  sourcesDrawer.classList.remove("hidden");
  sourcesBackdrop.classList.remove("hidden");
}
function closeSources(){
  sourcesDrawer.classList.add("hidden");
  sourcesBackdrop.classList.add("hidden");
}
btnCloseSources.addEventListener("click", closeSources);
sourcesBackdrop.addEventListener("click", closeSources);

setStatus("Session", "Ждет запуска", "idle");
interactionEnabled = false;
updateInteractionState();
updateModeUI();
connect();
updateEmptyState();
