const form = document.querySelector("#chat-form");
const input = document.querySelector("#message-input");
const messages = document.querySelector("#messages");
const sendButton = document.querySelector("#send-button");
const stopButton = document.querySelector("#stop-button");
const clearButton = document.querySelector("#clear-button");
const newSessionButton = document.querySelector("#new-session-button");
const sessionList = document.querySelector("#session-list");
const sessionCount = document.querySelector("#session-count");
const conversationTitle = document.querySelector("#conversation-title");
const turnInputTokens = document.querySelector("#turn-input-tokens");
const turnOutputTokens = document.querySelector("#turn-output-tokens");
const turnTotalTokens = document.querySelector("#turn-total-tokens");
const sessionTotalTokens = document.querySelector("#session-total-tokens");
const usageSource = document.querySelector("#usage-source");
const sessionId = localStorage.getItem("react-agent-session") || crypto.randomUUID();
localStorage.setItem("react-agent-session", sessionId);

let controller = null;
let activeSessionId = sessionId;
let streamingAssistant = null;
const emptyUsage = {
  prompt_tokens: 0,
  completion_tokens: 0,
  total_tokens: 0,
  estimated: false,
};

function updateUsage(usage, target) {
  const value = { ...emptyUsage, ...(usage || {}) };
  if (target === "turn") {
    turnInputTokens.textContent = value.prompt_tokens.toLocaleString();
    turnOutputTokens.textContent = value.completion_tokens.toLocaleString();
    turnTotalTokens.textContent = value.total_tokens.toLocaleString();
  } else {
    sessionTotalTokens.textContent = value.total_tokens.toLocaleString();
  }
  usageSource.textContent = value.estimated
    ? "Token 为估算值，模型服务未返回 usage"
    : "Token 来自模型 API usage";
}

function clearEmptyState() {
  messages.querySelector(".empty-state, .welcome-state")?.remove();
}

function appendMessage(role, content) {
  clearEmptyState();
  const row = document.createElement("div");
  row.className = `message-row ${role}`;
  const element = document.createElement("div");
  element.className = `message ${role}`;
  element.textContent = content;
  row.appendChild(element);
  messages.appendChild(row);
  messages.scrollTop = messages.scrollHeight;
  return element;
}

function appendAssistantDelta(content) {
  clearEmptyState();
  if (!streamingAssistant) {
    streamingAssistant = appendMessage("assistant", "");
  }
  streamingAssistant.textContent += content;
  messages.scrollTop = messages.scrollHeight;
}

function createSseParser(onEvent) {
  // 使用逐行状态机解析 SSE。网络 chunk 可以在任意字节位置截断，甚至把 \r\n
  // 拆到两个 chunk 中；逐行处理比按“双换行正则”切块更稳定，尤其适用于 Safari。
  let textBuffer = "";
  let eventType = "message";
  let dataLines = [];

  async function dispatchEvent() {
    if (!dataLines.length) {
      eventType = "message";
      return;
    }
    const rawData = dataLines.join("\n");
    const currentType = eventType;
    eventType = "message";
    dataLines = [];
    await onEvent({
      type: currentType,
      data: JSON.parse(rawData),
    });
  }

  async function processLine(line) {
    if (line === "") {
      await dispatchEvent();
      return;
    }
    if (line.startsWith(":")) return;
    const separator = line.indexOf(":");
    const field = separator === -1 ? line : line.slice(0, separator);
    let value = separator === -1 ? "" : line.slice(separator + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") eventType = value;
    if (field === "data") dataLines.push(value);
  }

  return {
    async push(text) {
      textBuffer += text;
      // SSE 允许 CRLF、单独 CR 或单独 LF。只处理已经拥有行结束符的内容，
      // 将最后一段保留到下一个网络 chunk，避免解析半行 JSON。
      const lines = textBuffer.split(/\r\n|\r|\n/);
      textBuffer = lines.pop() || "";
      for (const line of lines) {
        await processLine(line);
      }
    },
    async finish() {
      if (textBuffer) {
        await processLine(textBuffer);
        textBuffer = "";
      }
      await dispatchEvent();
    },
  };
}

function appendTool(data, finished = false) {
  clearEmptyState();
  const row = document.createElement("div");
  row.className = "message-row tool";
  const element = document.createElement("details");
  element.className = "tool-event";
  element.open = !finished;
  const summary = document.createElement("summary");
  summary.textContent = `${finished ? "工具完成" : "调用工具"} · ${data.name}`;
  const body = document.createElement("pre");
  body.textContent = JSON.stringify(data, null, 2);
  element.append(summary, body);
  row.appendChild(element);
  messages.appendChild(row);
  messages.scrollTop = messages.scrollHeight;
}

function resetMessages() {
  messages.replaceChildren();
  const empty = document.createElement("div");
  empty.className = "welcome-state";
  empty.innerHTML = [
    '<div class="welcome-icon" aria-hidden="true">✦</div>',
    "<h2>开始一段对话</h2>",
    "<p>输入问题，观察模型如何组织回答并调用工具。</p>",
  ].join("");
  messages.appendChild(empty);
}

function resetTurnUsage() {
  updateUsage(emptyUsage, "turn");
  usageSource.textContent = "等待第一轮对话";
}

function renderSessionList(sessions) {
  sessionList.replaceChildren();
  sessionCount.textContent = sessions.length.toLocaleString();
  if (!sessions.length) {
    const empty = document.createElement("div");
    empty.className = "session-empty";
    empty.textContent = "暂无会话";
    sessionList.appendChild(empty);
    return;
  }
  sessions.forEach((session) => {
    const item = document.createElement("div");
    item.className = `session-item${session.session_id === activeSessionId ? " active" : ""}`;
    item.dataset.sessionId = session.session_id;
    const selectButton = document.createElement("button");
    selectButton.type = "button";
    selectButton.className = "session-select";
    const title = document.createElement("strong");
    title.textContent = session.title || "新会话";
    const preview = document.createElement("span");
    preview.textContent = session.preview || "暂无消息";
    const total = document.createElement("small");
    total.textContent = `${session.usage.total_tokens.toLocaleString()} tokens`;
    selectButton.append(title, preview, total);
    selectButton.addEventListener("click", () => selectSession(session.session_id));

    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.className = "session-delete";
    deleteButton.title = "删除会话";
    deleteButton.setAttribute("aria-label", `删除会话：${session.title || "新会话"}`);
    deleteButton.textContent = "×";
    deleteButton.addEventListener("click", (event) => {
      event.stopPropagation();
      deleteSession(session.session_id);
    });

    item.append(selectButton, deleteButton);
    sessionList.appendChild(item);
    if (session.session_id === activeSessionId) {
      conversationTitle.textContent = session.title || "新会话";
    }
  });
}

async function deleteSession(sessionToDelete) {
  if (controller) return;
  if (!window.confirm("确定删除这个会话吗？删除后无法恢复。")) return;

  try {
    const response = await fetch(
      `/api/sessions/${encodeURIComponent(sessionToDelete)}`,
      { method: "DELETE" },
    );
    if (!response.ok) throw new Error(`删除失败：${response.status}`);

    if (sessionToDelete === activeSessionId) {
      activeSessionId = crypto.randomUUID();
      localStorage.setItem("react-agent-session", activeSessionId);
      conversationTitle.textContent = "新会话";
      resetMessages();
      resetTurnUsage();
      updateUsage(emptyUsage, "session");
    }
    await loadSessions();
    input.focus();
  } catch (error) {
    appendMessage("assistant", `删除会话失败：${error.message}`);
  }
}

async function loadSessions() {
  const response = await fetch("/api/sessions");
  if (!response.ok) throw new Error(`Failed to load sessions: ${response.status}`);
  renderSessionList(await response.json());
}

async function loadHistory() {
  const response = await fetch(`/api/sessions/${encodeURIComponent(activeSessionId)}`);
  if (!response.ok) throw new Error(`Failed to load session: ${response.status}`);
  const data = await response.json();
  resetMessages();
  data.messages.forEach((message) => appendMessage(message.role, message.content));
  resetTurnUsage();
  updateUsage(data.usage, "session");
  conversationTitle.textContent = data.messages.length
    ? (data.messages.find((message) => message.role === "user")?.content || "新会话")
    : "新会话";
}

async function selectSession(nextSessionId) {
  if (controller || nextSessionId === activeSessionId) return;
  activeSessionId = nextSessionId;
  localStorage.setItem("react-agent-session", activeSessionId);
  try {
    await loadHistory();
    await loadSessions();
  } catch (error) {
    appendMessage("assistant", `加载会话失败：${error.message}`);
  }
}

async function createSession() {
  if (controller) return;
  activeSessionId = crypto.randomUUID();
  localStorage.setItem("react-agent-session", activeSessionId);
  conversationTitle.textContent = "新会话";
  resetMessages();
  resetTurnUsage();
  updateUsage(emptyUsage, "session");
  await loadSessions();
  input.focus();
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message || controller) return;

  appendMessage("user", message);
  streamingAssistant = null;
  if (conversationTitle.textContent === "新会话") {
    conversationTitle.textContent = message.slice(0, 60);
  }
  input.value = "";
  updateUsage(emptyUsage, "turn");
  controller = new AbortController();
  sendButton.disabled = true;
  stopButton.hidden = false;

  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: activeSessionId, message }),
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(`Request failed: ${response.status}`);
    if (!response.headers.get("content-type")?.includes("text/event-stream")) {
      throw new Error("服务端没有返回 SSE 事件流");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    const handleEvent = async (eventData) => {
      if (!eventData || eventData.type === "connected") return;
      if (eventData.type === "tool_started") appendTool(eventData.data);
      if (eventData.type === "tool_finished") appendTool(eventData.data, true);
      if (eventData.type === "assistant_delta") {
        appendAssistantDelta(eventData.data.content || "");
      }
      if (eventData.type === "assistant_completed") {
        // 流式 delta 已经显示过时，只做最终内容校正，避免重复追加。
        if (streamingAssistant) {
          streamingAssistant.textContent = eventData.data.content || "";
        } else {
          appendMessage("assistant", eventData.data.content);
        }
      }
      if (eventData.type === "model_usage") {
        // Agent 每调用一次模型就会发送一次，实时显示当前 run 的累计值。
        updateUsage(eventData.data, "turn");
      }
      if (eventData.type === "done") {
        updateUsage(eventData.data.usage, "turn");
        updateUsage(eventData.data.session_usage, "session");
        await loadSessions();
      }
      if (eventData.type === "error") {
        appendMessage("assistant", `错误：${eventData.data.message}`);
      }
    };
    const sseParser = createSseParser(handleEvent);
    while (true) {
      const { value, done } = await reader.read();
      if (value) {
        await sseParser.push(decoder.decode(value, { stream: true }));
      }
      if (done) {
        await sseParser.push(decoder.decode());
        break;
      }
    }
    await sseParser.finish();
  } catch (error) {
    if (error.name !== "AbortError") appendMessage("assistant", `请求失败：${error.message}`);
  } finally {
    streamingAssistant = null;
    controller = null;
    sendButton.disabled = false;
    stopButton.hidden = true;
    input.focus();
  }
});

stopButton.addEventListener("click", () => controller?.abort());

clearButton.addEventListener("click", async () => {
  controller?.abort();
  await fetch(`/api/sessions/${encodeURIComponent(activeSessionId)}`, { method: "DELETE" });
  resetMessages();
  resetTurnUsage();
  updateUsage(emptyUsage, "session");
  await loadSessions();
});

newSessionButton.addEventListener("click", createSession);

document.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "n") {
    event.preventDefault();
    createSession();
  }
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

async function initialize() {
  try {
    await loadSessions();
    await loadHistory();
  } catch (error) {
    appendMessage("assistant", `加载会话失败：${error.message}`);
  }
}

initialize();
