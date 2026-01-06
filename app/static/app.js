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

const feed = $("feed");
const feedCard = $("feedCard");
const events = $("events");
const meta = $("meta");
const hint = $("hint");

const btnStart = $("btnStart");
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
let availableSources = [];
let selectedSources = [];

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

function addMathStyling(html){
  if (!html) return "";
  const blockRe = /\$\$(.+?)\$\$/gs;
  const inlineRe = /(^|[^\\])\$(.+?)\$/g;
  let out = html.replace(blockRe, (_, expr) => `<div class="math block">${expr}</div>`);
  out = out.replace(inlineRe, (_, lead, expr) => `${lead}<span class="math">${expr}</span>`);
  return out;
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
    agentState === "running" ? "rgba(34,197,94,.16)" :
    agentState === "error" ? "rgba(239,68,68,.18)" :
    "rgba(148,163,184,.16)";

  if (typeof counter === "number"){
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
  btnToggleNotes.textContent = "Показать конспект";
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
    btnToggleNotes.textContent = "Показать конспект";
  }
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
    btnToggleNotes.textContent = "Скрыть конспект";
  } else {
    qNotes.classList.add("hidden");
    btnToggleNotes.textContent = "Показать конспект";
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
      availableSources = msg.sources || [];
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

    if (msg.type === "message"){
      if (msg.role === "user"){
        addUserAnswer(msg.body || "");
      } else if (msg.role === "system"){
        addAnalysis(msg.title || "Система", msg.body || "");
      }
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
    ws?.send(JSON.stringify({type:"answer", text}));
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

btnNew.addEventListener("click", () => {
  feed.innerHTML = "";
  events.innerHTML = "";
  showPrestartCard();
  ws.send(JSON.stringify({type:"reset"}));
  updateEmptyState();
  interactionEnabled = false;
  updateInteractionState();
  btnStart.disabled = false;
  closeSources();
});

btnStart.addEventListener("click", () => {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  hasStarted = true;
  setStatus("Session", "Подключение…", "running");
  setQuestionLoading("Server", "Готовлю вопрос…");
  ws.send(JSON.stringify({type:"start", sources: selectedSources}));
  btnStart.disabled = true;
  closeSources();
});

function updateInteractionState(){
  const enabled = interactionEnabled && !analysisLocked;
  input.disabled = !enabled;
  updateSendClearDisabled();
  updateMicDisabled();
  if (!interactionEnabled){
    btnMic.classList.remove("listening", "loading");
  }
}

function updateSendClearDisabled(){
  const shouldDisable = !interactionEnabled || analysisLocked || recording || transcribing;
  btnSend.disabled = shouldDisable;
  btnClear.disabled = shouldDisable;
}

function updateMicDisabled(){
  const shouldDisable = !sttEnabled || !interactionEnabled || analysisLocked;
  btnMic.disabled = shouldDisable;
}

function renderSourcePicker(){
  if (!sourcePicker) return;
  sourcePicker.innerHTML = "";
  sourcePicker.classList.add("sourcePicker");
  if (!availableSources.length){
    sourcePicker.classList.add("hidden");
    return;
  }
  sourcePicker.classList.remove("hidden");
  const list = document.createElement("div");
  list.className = "sourceList";

  availableSources.forEach((src) => {
    const id = `src_${src.replace(/[^a-zA-Z0-9_-]/g, "_")}`;
    const label = document.createElement("label");
    label.className = "sourceItem";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.id = id;
    cb.value = src;
    cb.checked = selectedSources.includes(src);
    cb.addEventListener("change", () => {
      if (cb.checked){
        if (!selectedSources.includes(src)) selectedSources.push(src);
      } else {
        selectedSources = selectedSources.filter((s) => s !== src);
      }
    });
    label.appendChild(cb);
    const span = document.createElement("span");
    span.textContent = src;
    label.appendChild(span);
    list.appendChild(label);
  });
  sourcePicker.appendChild(list);
}

function showPrestartCard(){
  hasStarted = false;
  qTitle.textContent = "Ожидаю запуска";
  qMeta.textContent = "";
  qProgress.textContent = "";
  qBody.innerHTML = "";
  qNotes.classList.add("hidden");
  btnToggleNotes.textContent = "Выбор тем";
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
showPrestartCard();
connect();
updateEmptyState();
