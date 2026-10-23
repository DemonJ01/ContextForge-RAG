"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const STORAGE_KEY = "rag.workspace.v1";
  const SVG_NS = "http://www.w3.org/2000/svg";
  const state = {
    conversations: [], activeId: null, draft: "", scope: "reference", view: "chat",
    health: null, healthError: null, documents: [], documentsLoaded: false,
    documentsError: null, pending: null, evidence: null, overlay: null,
  };
  let toastTimer;
  let documentController;
  let overlayTrigger;

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function icon(name) {
    const svg = document.createElementNS(SVG_NS, "svg");
    svg.classList.add("icon");
    svg.setAttribute("aria-hidden", "true");
    const use = document.createElementNS(SVG_NS, "use");
    use.setAttribute("href", `#i-${name}`);
    svg.append(use);
    return svg;
  }

  function action(text, handler, className = "answer-action", iconName) {
    const button = element("button", className);
    button.type = "button";
    if (iconName) button.append(icon(iconName));
    button.append(element("span", "", text));
    button.addEventListener("click", handler);
    return button;
  }

  function uniqueId() {
    return typeof crypto.randomUUID === "function" ? crypto.randomUUID() : `rag-${Date.now()}-${Math.random().toString(36).slice(2)}`;
  }

  function announce(message) { $("request-status").textContent = message; }

  function notify(message) {
    clearTimeout(toastTimer);
    $("toast").textContent = message;
    $("toast").hidden = false;
    toastTimer = setTimeout(() => { $("toast").hidden = true; }, 3500);
  }

  function persist() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({
        conversations: state.conversations.slice(0, 30), activeId: state.activeId,
      }));
    } catch (_) {
      notify("Browser storage is unavailable. This conversation will last for this visit.");
    }
  }

  function restore() {
    try {
      const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
      if (!saved || !Array.isArray(saved.conversations)) return;
      state.conversations = saved.conversations.filter((c) => c && typeof c.id === "string"
        && typeof c.title === "string" && Array.isArray(c.messages)).slice(0, 30).map((c) => ({
          ...c, messages: c.messages.filter((m) => m && ["user", "answer", "error"].includes(m.role)
            && typeof m.content === "string"),
        }));
      state.activeId = state.conversations.some((c) => c.id === saved.activeId) ? saved.activeId : null;
      const conversation = activeConversation();
      if (conversation) state.scope = conversation.scope === "all" ? "all" : "reference";
    } catch (_) { /* A malformed local history should never prevent startup. */ }
  }

  function activeConversation() {
    return state.conversations.find((c) => c.id === state.activeId) || null;
  }

  function isCloud(mode = state.health?.mode) { return mode === "cloud"; }
  function modeName(mode = state.health?.mode) { return isCloud(mode) ? "Cloud · remote index" : "Local · extractive"; }
  function collectionName(type) { return type === "true" ? "Reference" : "Extended library"; }

  async function request(path, options = {}) {
    const response = await fetch(path, {
      headers: { "Accept": "application/json", ...(options.body ? { "Content-Type": "application/json" } : {}) },
      ...options,
    });
    let data;
    try { data = await response.json(); }
    catch (_) { throw new Error("The workspace returned an unreadable response. Please try again."); }
    if (!response.ok) {
      const detail = typeof data.detail === "string" ? data.detail.slice(0, 500)
        : response.status === 422 ? "Keep your question between 1 and 2,000 characters, then try again."
          : "The workspace could not complete this request. Please try again.";
      throw new Error(detail);
    }
    return data;
  }

  function errorText(error) {
    if (error?.name === "AbortError") return "The request took too long. Please try again with a shorter question.";
    if (error instanceof TypeError) return "The workspace is unreachable. Check your connection and that the RAG server is running.";
    return error?.message || "Something interrupted the search. Please try again.";
  }

  function updateControls() {
    const pending = Boolean(state.pending);
    $("send-button").disabled = pending || !$("question").value.trim();
    $("send-button").setAttribute("aria-label", pending ? "Search in progress" : "Send question");
    $("query-form").setAttribute("aria-busy", String(pending));
    $("scope").disabled = pending;
    $("scope-help").textContent = isCloud()
      ? state.scope === "all" ? "Searching your entire remote index. The library contains bundled documents."
        : "Searching reference documents in your remote index. The library contains bundled documents."
      : state.scope === "all" ? "Searching reference and extended library documents."
        : "Answers begin with your reference documents.";
    document.querySelectorAll(".suggestion").forEach((button) => { button.disabled = pending; });
    document.querySelectorAll(".error-message button").forEach((button) => { button.disabled = pending; });
  }

  function renderHistory() {
    const list = $("conversation-list");
    list.replaceChildren();
    state.conversations.forEach((conversation) => {
      const button = action(conversation.title, () => selectConversation(conversation.id), "history-item", "chat");
      button.title = conversation.title;
      if (conversation.id === state.activeId && state.view === "chat") {
        button.classList.add("active");
        button.setAttribute("aria-current", "page");
      }
      list.append(button);
    });
    $("history-empty").hidden = state.conversations.length > 0;
    $("clear-history").hidden = state.conversations.length === 0;
    $("clear-history").disabled = Boolean(state.pending);
  }

  function showView(view) {
    state.view = view;
    const library = view === "library";
    $("chat-view").hidden = library;
    $("library-view").hidden = !library;
    $("top-library").querySelector("use").setAttribute("href", library ? "#i-chat" : "#i-book");
    $("top-library").querySelector("span").textContent = library ? "Conversation" : "Library";
    $("top-library").setAttribute("aria-label", library ? "Return to conversation" : "Open knowledge library");
    $("chat-nav").classList.toggle("active", !library);
    $("library-nav").classList.toggle("active", library);
    $("chat-nav").toggleAttribute("aria-current", !library);
    $("library-nav").toggleAttribute("aria-current", library);
    if (!library) $("chat-nav").setAttribute("aria-current", "page");
    else $("library-nav").setAttribute("aria-current", "page");
    $("breadcrumb-current").textContent = library ? "Knowledge library" : activeConversation()?.title || "New conversation";
    closeSidebar();
    closeEvidence();
    renderHistory();
    if (library) {
      renderLibrary();
      $("library-heading").tabIndex = -1;
      $("library-heading").focus({ preventScroll: true });
    }
  }

  function newConversation() {
    state.activeId = null;
    state.draft = "";
    state.scope = "reference";
    $("question").value = "";
    $("scope").value = "reference";
    resizeComposer();
    showView("chat");
    renderConversation();
    persist();
    $("question").focus();
  }

  function selectConversation(id) {
    state.activeId = id;
    const conversation = activeConversation();
    state.scope = conversation?.scope === "all" ? "all" : "reference";
    $("scope").value = state.scope;
    $("question").value = "";
    resizeComposer();
    showView("chat");
    renderConversation();
    persist();
  }

  function normalizedCitations(message) {
    if (Array.isArray(message.citations) && message.citations.length) return message.citations.filter((c) => c && typeof c === "object");
    if (Array.isArray(message.sources)) return message.sources.filter((s) => typeof s === "string").map((source, index) => ({
      id: `context-${index + 1}`, title: `Retrieved passage ${index + 1}`,
      excerpt: source, legacy: true,
    }));
    return [];
  }

  function appendAnswerText(container, text, citations) {
    String(text).split(/\n\s*\n/).filter(Boolean).forEach((paragraph) => {
      const p = element("p");
      const pattern = /\[(\d+)\]/g;
      let position = 0;
      let match;
      while ((match = pattern.exec(paragraph))) {
        p.append(document.createTextNode(paragraph.slice(position, match.index)));
        const index = Number(match[1]) - 1;
        if (citations[index]) {
          const citation = action(match[1].padStart(2, "0"), () => openEvidence(citations, index), "inline-citation");
          citation.setAttribute("aria-label", `View source ${match[1]}: ${citations[index].title || "Retrieved passage"}`);
          p.append(citation);
        } else p.append(document.createTextNode(match[0]));
        position = pattern.lastIndex;
      }
      p.append(document.createTextNode(paragraph.slice(position)));
      container.append(p);
    });
  }

  function renderUser(message) {
    const article = element("article", "message message-user");
    article.append(element("div", "message-eyebrow", "YOUR QUESTION"), element("div", "message-content", message.content));
    return article;
  }

  function renderAnswer(message) {
    const article = element("article", "message message-answer");
    const heading = element("div", "message-eyebrow");
    const wordmark = element("span", "answer-wordmark");
    wordmark.append(icon("layers"), document.createTextNode("RAG"));
    heading.append(wordmark, element("span", "answer-mode", modeName(message.mode).toUpperCase()));
    const content = element("div", "answer-content");
    const citations = normalizedCitations(message);
    appendAnswerText(content, message.content, citations);
    const actions = element("div", "answer-actions");
    if (citations.length) actions.append(action(`${citations.length} source${citations.length === 1 ? "" : "s"}`, () => openEvidence(citations), "answer-action", "book"));
    actions.append(action("Copy answer", async () => {
      try {
        await navigator.clipboard.writeText(message.content);
        notify("Answer copied to clipboard.");
      } catch (_) { notify("Clipboard access is unavailable. Select the answer text to copy it."); }
    }, "answer-action", "copy"));
    if (Number.isFinite(message.elapsed_ms)) {
      const timing = message.elapsed_ms >= 1000 ? `${(message.elapsed_ms / 1000).toFixed(1)} s` : `${Math.round(message.elapsed_ms)} ms`;
      actions.append(element("span", "answer-time", timing));
    }
    article.append(heading, content, actions);
    return article;
  }

  function renderError(message, conversation) {
    const article = element("article", "error-message");
    article.setAttribute("role", "alert");
    article.append(element("h2", "", "We couldn’t finish that search."), element("p", "", message.content));
    const retry = action("Try again", () => {
      if (state.pending) return;
      state.activeId = conversation.id;
      showView("chat");
      sendQuestion(message.question, message.scope, message.id);
    });
    retry.disabled = Boolean(state.pending);
    article.append(retry);
    return article;
  }

  function renderPending() {
    const article = element("article", "message pending-message");
    article.setAttribute("aria-busy", "true");
    const heading = element("div", "message-eyebrow");
    heading.append(element("span", "loading-indicator"), document.createTextNode(isCloud() ? "SEARCHING THE REMOTE INDEX" : "READING YOUR SOURCES"));
    article.append(heading, element("p", "", "Finding the passages that speak to your question."));
    const lines = element("div", "pending-lines");
    lines.setAttribute("aria-hidden", "true");
    lines.append(element("span"), element("span"), element("span"));
    article.append(lines);
    return article;
  }

  function renderConversation(scrollToBottom = true) {
    const conversation = activeConversation();
    const hasMessages = Boolean(conversation?.messages.length);
    $("empty-state").hidden = hasMessages;
    $("messages").hidden = !hasMessages;
    const messages = $("messages");
    messages.replaceChildren();
    if (conversation) {
      const pendingIndex = state.pending?.conversationId === conversation.id ? state.pending.messageIndex : -1;
      conversation.messages.forEach((message, index) => {
        if (index === pendingIndex) messages.append(renderPending());
        if (message.role === "user") messages.append(renderUser(message));
        else if (message.role === "answer") messages.append(renderAnswer(message));
        else if (message.role === "error") messages.append(renderError(message, conversation));
      });
      if (pendingIndex === conversation.messages.length) messages.append(renderPending());
    }
    if (state.view === "chat") $("breadcrumb-current").textContent = conversation?.title || "New conversation";
    updateControls();
    renderHistory();
    if (scrollToBottom && hasMessages) requestAnimationFrame(() => {
      $("conversation-scroll").scrollTop = $("conversation-scroll").scrollHeight;
    });
  }

  async function sendQuestion(question, scope = state.scope, retryId = null) {
    const trimmed = String(question || "").trim();
    if (!trimmed || state.pending) return;
    if (trimmed.length > 2000) { notify("Keep your question to 2,000 characters or fewer."); $("question").focus(); return; }
    let conversation = activeConversation();
    if (!conversation) {
      conversation = { id: uniqueId(), title: trimmed.slice(0, 90), scope, messages: [], createdAt: Date.now() };
      state.conversations.unshift(conversation);
      state.conversations = state.conversations.slice(0, 30);
      state.activeId = conversation.id;
    }
    conversation.scope = scope;
    state.scope = scope;
    $("scope").value = scope;
    state.conversations = [conversation, ...state.conversations.filter((item) => item.id !== conversation.id)];
    let messageIndex;
    if (retryId) {
      messageIndex = conversation.messages.findIndex((m) => m.id === retryId);
      if (messageIndex < 0) return;
      conversation.messages.splice(messageIndex, 1);
    }
    else conversation.messages.push({ id: uniqueId(), role: "user", content: trimmed });
    messageIndex ??= conversation.messages.length;
    $("question").value = "";
    resizeComposer();
    state.pending = { conversationId: conversation.id, question: trimmed, messageIndex };
    showView("chat");
    renderConversation();
    persist();
    announce("Searching your documents. The answer will appear when the search is complete.");
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 120000);
    try {
      const payload = { q: trimmed, thread_id: conversation.id, scope, top_k: 4 };
      const response = await request("/query", { method: "POST", body: JSON.stringify(payload), signal: controller.signal });
      if (typeof response.answer !== "string" || !response.answer.trim() || response.status === "error") {
        throw new Error("The search returned no answer. Please try again or ask a more specific question.");
      }
      const message = { ...response, id: uniqueId(), role: "answer", content: response.answer };
      conversation.messages.splice(messageIndex, 0, message);
      announce("Answer ready. Review the answer and its sources below.");
      state.pending = null;
      if (state.activeId === conversation.id && state.view === "chat") {
        renderConversation();
        const citations = normalizedCitations(message);
        if (citations.length && window.matchMedia("(min-width: 1101px)").matches) openEvidence(citations);
      }
    } catch (error) {
      conversation.messages.splice(messageIndex, 0, { id: uniqueId(), role: "error", content: errorText(error), question: trimmed, scope });
      state.pending = null;
      if (state.activeId === conversation.id && state.view === "chat") renderConversation();
      announce("The search could not be completed. A retry option is available in the conversation.");
    } finally {
      clearTimeout(timer);
      state.pending = null;
      persist();
      updateControls();
      renderHistory();
    }
  }

  function renderConnection() {
    const connected = Boolean(state.health);
    $("connection-dot").className = `status-dot ${connected ? "connected" : "disconnected"}`;
    const topDot = $("top-mode").querySelector(".status-dot");
    topDot.className = `status-dot ${connected ? "connected" : "disconnected"}`;
    $("connection-label").textContent = connected ? modeName() : "Workspace unavailable";
    $("connection-detail").textContent = connected ? "Connected to your workspace" : "Check connection details";
    $("top-mode-text").textContent = connected ? isCloud() ? "Remote index" : "Local search" : "Offline";
    $("connection-description").textContent = connected
      ? isCloud() ? "Answers use your configured cloud models and remote retrieval index. The knowledge library shows the documents bundled with this workspace."
        : "Search runs locally across the indexed document text. Answers are extracted source passages; no language model or cloud credentials are used."
      : state.healthError || "The workspace could not be reached. Check that the server is running and try again.";
    const stats = $("connection-stats");
    stats.replaceChildren();
    if (connected) {
      const rows = [
        ["Answer mode", isCloud() ? "Cloud" : "Extractive"],
        [isCloud() ? "Bundled documents" : "Indexed documents", state.health.document_count],
        ["Reference documents", state.health.reference_document_count ?? state.health.reference_count],
        ["Indexed passages", state.health.chunk_count],
      ];
      rows.forEach(([label, value]) => {
        if (value === undefined || value === null) return;
        const row = element("div");
        row.append(element("dt", "", label), element("dd", "", value));
        stats.append(row);
      });
      $("library-count").textContent = Number.isFinite(state.health.document_count) ? state.health.document_count : "—";
    }
    $("library-heading").textContent = isCloud() ? "Your bundled library." : "The source of it all.";
    updateControls();
  }

  async function checkHealth() {
    $("refresh-connection").disabled = true;
    try {
      state.health = await request("/health");
      state.healthError = null;
    } catch (error) {
      state.health = null;
      state.healthError = errorText(error);
    } finally {
      $("refresh-connection").disabled = false;
      renderConnection();
    }
  }

  async function loadDocuments() {
    state.documentsError = null;
    state.documentsLoaded = false;
    renderLibrary();
    try {
      const response = await request("/documents?scope=all");
      if (!Array.isArray(response.documents)) throw new Error("The document library returned an unreadable response.");
      state.documents = response.documents.filter((d) => d && typeof d.id === "string");
      state.documents.sort((a, b) => Number(b.source_type === "true") - Number(a.source_type === "true")
        || String(a.title).localeCompare(String(b.title)));
      state.documentsLoaded = true;
      $("library-count").textContent = state.documents.length;
    } catch (error) {
      state.documentsError = errorText(error);
    }
    renderLibrary();
  }

  function renderLibrary() {
    const list = $("document-list");
    list.replaceChildren();
    if (!state.documentsLoaded && !state.documentsError) {
      list.setAttribute("aria-busy", "true");
      const skeleton = element("div", "library-loading");
      skeleton.setAttribute("aria-label", "Loading documents");
      skeleton.append(element("span"), element("span"), element("span"));
      list.append(skeleton);
      $("library-total").textContent = "Loading library";
      return;
    }
    list.setAttribute("aria-busy", "false");
    if (state.documentsError) {
      const error = element("div", "library-empty");
      error.append(icon("book"), element("h2", "", "The library is out of reach."), element("p", "", state.documentsError), action("Try again", loadDocuments, "secondary-button", "refresh"));
      list.append(error);
      $("library-total").textContent = "Connection unavailable";
      return;
    }
    const query = $("document-search").value.trim().toLowerCase();
    const filter = $("document-filter").value;
    const documents = state.documents.filter((doc) => (!query || `${doc.title} ${doc.filename}`.toLowerCase().includes(query))
      && (filter === "all" || (filter === "reference" ? doc.source_type === "true" : doc.source_type !== "true")));
    $("library-total").textContent = `${documents.length}${documents.length === state.documents.length ? "" : ` of ${state.documents.length}`} document${documents.length === 1 ? "" : "s"}`;
    if (!documents.length) {
      const empty = element("div", "library-empty");
      empty.append(icon("search"), element("h2", "", "A different starting point?"), element("p", "", "No documents match these filters. Try a title, topic, or a different collection."));
      empty.append(action("Reset filters", () => { $("document-search").value = ""; $("document-filter").value = "all"; renderLibrary(); }, "secondary-button"));
      list.append(empty);
      return;
    }
    documents.forEach((doc) => {
      const row = element("button", "document-row");
      row.type = "button";
      row.setAttribute("aria-label", `Read ${doc.title || doc.filename}`);
      const identity = element("span", "document-identity");
      const format = String(doc.format || "text").toUpperCase();
      const name = element("span");
      name.append(element("span", "document-name", doc.title || doc.filename), element("span", "document-filename", doc.filename));
      identity.append(element("span", "document-format", format), name);
      row.append(identity, element("span", "document-collection", collectionName(doc.source_type)), element("span", "document-chunks", doc.chunk_count ?? "—"), icon("arrow"));
      row.addEventListener("click", () => openDocument(doc.id));
      list.append(row);
    });
  }

  async function openDocument(id, excerpt = "") {
    closeEvidence();
    if (documentController) documentController.abort();
    documentController = new AbortController();
    const controller = documentController;
    const dialog = $("document-dialog");
    $("document-title").textContent = "Opening your source…";
    $("document-metadata").textContent = "";
    $("document-content").replaceChildren(element("span", "loading-indicator"));
    $("document-content").setAttribute("aria-busy", "true");
    if (!dialog.open) dialog.showModal();
    try {
      const doc = await request(`/documents/${encodeURIComponent(id)}`, { signal: controller.signal });
      if (documentController !== controller) return;
      $("document-title").textContent = doc.title || doc.filename || "Source document";
      $("document-kicker").textContent = `${collectionName(doc.source_type).toUpperCase()} / SOURCE DOCUMENT`;
      $("document-metadata").textContent = [doc.filename, doc.format ? String(doc.format).toUpperCase() : null,
        Number.isFinite(doc.chunk_count) ? `${doc.chunk_count} indexed passages` : null].filter(Boolean).join(" · ");
      const content = $("document-content");
      content.replaceChildren();
      if (excerpt) content.append(element("div", "document-highlight", excerpt));
      content.append(element("div", "", typeof doc.content === "string" && doc.content.trim() ? doc.content : "No extracted text is available for this document."));
      content.scrollTop = 0;
    } catch (error) {
      if (error.name === "AbortError") return;
      $("document-title").textContent = "This source couldn’t be opened.";
      const content = $("document-content");
      content.replaceChildren(element("p", "", errorText(error)), action("Try again", () => openDocument(id, excerpt), "secondary-button", "refresh"));
    } finally {
      if (documentController === controller) $("document-content").setAttribute("aria-busy", "false");
    }
  }

  function openEvidence(citations, selectedIndex = null) {
    closeSidebar();
    state.evidence = citations;
    const panel = $("evidence-panel");
    const list = $("evidence-list");
    list.replaceChildren();
    $("evidence-description").textContent = `${citations.length} retrieved passage${citations.length === 1 ? "" : "s"} behind this answer.`;
    citations.forEach((citation, index) => {
      const card = element("article", "evidence-card");
      card.id = `evidence-source-${index}`;
      card.tabIndex = -1;
      const top = element("div", "evidence-card-top");
      top.append(element("span", "evidence-number", String(index + 1).padStart(2, "0")), element("span", "evidence-type", citation.legacy ? "RETRIEVED CONTEXT" : collectionName(citation.source_type).toUpperCase()));
      const title = citation.title || citation.filename || `Source ${index + 1}`;
      const titleNode = citation.document_id
        ? action(title, () => openDocument(citation.document_id, citation.excerpt), "evidence-title")
        : element("p", "evidence-title", title);
      const excerpt = typeof citation.excerpt === "string" ? citation.excerpt : "";
      card.append(top, titleNode, element("p", "evidence-excerpt", excerpt));
      if (citation.document_id) card.append(action("Read document", () => openDocument(citation.document_id, excerpt), "evidence-open", "arrow"));
      list.append(card);
    });
    panel.hidden = false;
    $("conversation-layout").classList.add("has-evidence");
    if (window.matchMedia("(max-width: 1100px)").matches) {
      state.overlay = "evidence";
      overlayTrigger = document.activeElement;
      $("evidence-scrim").hidden = false;
      panel.setAttribute("role", "dialog");
      panel.setAttribute("aria-modal", "true");
      // Keep the evidence panel active while the surrounding workspace is inert.
      $("conversation-scroll").inert = true;
      $("query-form").inert = true;
      $("sidebar").inert = true;
      $("main-content").querySelector(".topbar").inert = true;
      $("close-evidence").focus();
    }
    if (selectedIndex !== null) requestAnimationFrame(() => {
      const card = $(`evidence-source-${selectedIndex}`);
      card?.scrollIntoView({ block: "nearest", behavior: "auto" });
      card?.focus({ preventScroll: true });
    });
  }

  function closeEvidence() {
    $("evidence-panel").hidden = true;
    $("evidence-scrim").hidden = true;
    $("conversation-layout").classList.remove("has-evidence");
    $("evidence-panel").removeAttribute("role");
    $("evidence-panel").removeAttribute("aria-modal");
    $("conversation-scroll").inert = false;
    $("query-form").inert = false;
    $("sidebar").inert = false;
    $("main-content").querySelector(".topbar").inert = false;
    if (state.overlay === "evidence") {
      state.overlay = null;
      if (overlayTrigger?.isConnected) overlayTrigger.focus({ preventScroll: true });
    }
  }

  function openSidebar() {
    if (window.matchMedia("(min-width: 769px)").matches) return;
    closeEvidence();
    state.overlay = "sidebar";
    overlayTrigger = document.activeElement;
    $("sidebar").classList.add("is-open");
    $("sidebar").setAttribute("role", "dialog");
    $("sidebar").setAttribute("aria-modal", "true");
    $("sidebar-scrim").hidden = false;
    $("mobile-menu").setAttribute("aria-expanded", "true");
    $("main-content").inert = true;
    $("sidebar-close").focus();
  }

  function closeSidebar() {
    $("sidebar").classList.remove("is-open");
    $("sidebar").removeAttribute("role");
    $("sidebar").removeAttribute("aria-modal");
    $("sidebar-scrim").hidden = true;
    $("mobile-menu").setAttribute("aria-expanded", "false");
    $("main-content").inert = false;
    if (state.overlay === "sidebar") {
      state.overlay = null;
      if (overlayTrigger?.isConnected) overlayTrigger.focus({ preventScroll: true });
    }
  }

  function resizeComposer() {
    const textarea = $("question");
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(Math.max(48, textarea.scrollHeight), 180)}px`;
    updateControls();
  }

  function bindEvents() {
    $("query-form").addEventListener("submit", (event) => { event.preventDefault(); sendQuestion($("question").value); });
    $("question").addEventListener("input", resizeComposer);
    $("question").addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        if (!$("send-button").disabled) sendQuestion($("question").value);
      }
    });
    $("scope").addEventListener("change", () => {
      state.scope = $("scope").value;
      const conversation = activeConversation();
      if (conversation) { conversation.scope = state.scope; persist(); }
      updateControls();
    });
    $("suggestions").addEventListener("click", (event) => {
      const suggestion = event.target.closest("[data-question]");
      if (suggestion && !suggestion.disabled) sendQuestion(suggestion.dataset.question);
    });
    $("new-chat").addEventListener("click", newConversation);
    $("chat-nav").addEventListener("click", () => { showView("chat"); renderConversation(false); });
    $("library-nav").addEventListener("click", () => showView("library"));
    $("top-library").addEventListener("click", () => {
      const view = state.view === "library" ? "chat" : "library";
      showView(view);
      // A query can finish while the library is visible. Render the saved
      // result when returning, rather than leaving its old loading state.
      if (view === "chat") renderConversation(false);
    });
    $("document-search").addEventListener("input", renderLibrary);
    $("document-filter").addEventListener("change", renderLibrary);
    $("close-document").addEventListener("click", () => $("document-dialog").close());
    $("document-dialog").addEventListener("close", () => { documentController?.abort(); });
    $("connection-button").addEventListener("click", () => { closeSidebar(); $("connection-dialog").showModal(); });
    $("close-connection").addEventListener("click", () => $("connection-dialog").close());
    $("refresh-connection").addEventListener("click", () => { checkHealth(); if (state.documentsError) loadDocuments(); });
    $("clear-history").addEventListener("click", () => { closeSidebar(); $("clear-dialog").showModal(); });
    $("close-clear").addEventListener("click", () => $("clear-dialog").close());
    $("cancel-clear").addEventListener("click", () => $("clear-dialog").close());
    $("confirm-clear").addEventListener("click", () => {
      state.conversations = [];
      $("clear-dialog").close();
      newConversation();
      notify("Saved conversations cleared.");
    });
    $("mobile-menu").addEventListener("click", openSidebar);
    $("sidebar-close").addEventListener("click", closeSidebar);
    $("sidebar-scrim").addEventListener("click", closeSidebar);
    $("close-evidence").addEventListener("click", closeEvidence);
    $("evidence-scrim").addEventListener("click", closeEvidence);
    document.querySelectorAll("dialog").forEach((dialog) => {
      dialog.addEventListener("click", (event) => {
        if (event.target !== dialog) return;
        const bounds = dialog.getBoundingClientRect();
        if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) dialog.close();
      });
    });
    document.addEventListener("keydown", (event) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        if (document.querySelector("dialog[open]")) return;
        event.preventDefault(); newConversation();
      }
      if (event.key === "Escape") {
        if (state.overlay === "sidebar") closeSidebar();
        else if (state.overlay === "evidence") closeEvidence();
      }
      if (event.key === "Tab" && state.overlay) {
        const overlay = state.overlay === "sidebar" ? $("sidebar") : $("evidence-panel");
        const controls = Array.from(overlay.querySelectorAll("button:not(:disabled), a[href], select, [tabindex='0']"))
          .filter((node) => node.getClientRects().length > 0);
        if (!controls.length) return;
        const first = controls[0]; const last = controls[controls.length - 1];
        if (event.shiftKey && (document.activeElement === first || !overlay.contains(document.activeElement))) {
          event.preventDefault(); last.focus();
        } else if (!event.shiftKey && (document.activeElement === last || !overlay.contains(document.activeElement))) {
          event.preventDefault(); first.focus();
        }
      }
    });
    window.matchMedia("(min-width: 769px)").addEventListener("change", (event) => { if (event.matches) closeSidebar(); });
    window.matchMedia("(max-width: 1100px)").addEventListener("change", () => { closeEvidence(); });
  }

  restore();
  $("scope").value = state.scope;
  bindEvents();
  renderConversation();
  Promise.allSettled([checkHealth(), loadDocuments()]);
})();
