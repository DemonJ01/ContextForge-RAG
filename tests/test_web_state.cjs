"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");
const { webcrypto } = require("node:crypto");

const source = fs.readFileSync(path.join(__dirname, "../ui/web/rag.js"), "utf8");
const startup = source.lastIndexOf("\n  restore();");
assert(startup > 0, "The test harness must find the application startup boundary.");
// Keep the actual draft, navigation, submission, storage, and event functions.
// Rendering is omitted because these state checks do not run in a browser.
const stateSource = source.slice(0, startup) + `
  showView = (view) => { state.view = view; };
  renderConversation = () => { updateControls(); };
  renderHistory = () => {};
  checkHealth = async () => {};
  globalThis.app = { state, persist, restore, restoreDraft, selectConversation,
    newConversation, sendQuestion, bindEvents };
})();`;

function harness(storage = { value: null }, fetch = async () => answer()) {
  const elements = new Map();
  const timers = new Map();
  let timerId = 0;
  function element(id) {
    if (!elements.has(id)) elements.set(id, {
      value: "", style: {}, scrollHeight: 48, listeners: {},
      setAttribute() {}, focus() {},
      addEventListener(event, listener) { this.listeners[event] = listener; },
    });
    return elements.get(id);
  }
  const context = {
    crypto: webcrypto, AbortController, TypeError, fetch,
    localStorage: {
      getItem() { return storage.value; },
      setItem(_, value) { storage.value = value; },
    },
    setTimeout(callback) { const id = ++timerId; timers.set(id, callback); return id; },
    clearTimeout(id) { timers.delete(id); },
    document: { getElementById: element, querySelectorAll: () => [], addEventListener() {} },
    window: { matchMedia: () => ({ matches: false, addEventListener() {} }), addEventListener() {} },
  };
  vm.createContext(context);
  vm.runInContext(stateSource, context);
  context.app.state.health = { status: "ready", mode: "local" };
  context.app.bindEvents();
  function type(value) {
    element("question").value = value;
    element("question").listeners.input();
  }
  function reload() {
    const restored = harness(storage, fetch);
    restored.app.restore();
    restored.app.restoreDraft();
    return restored;
  }
  return { app: context.app, element, type, storage, reload, timers };
}

function conversation(id, scope = "reference") {
  return { id, title: `Conversation ${id}`, scope, draft: "", messages: [] };
}

function answer() {
  return { ok: true, json: async () => ({ answer: "A supported answer.", status: "complete", citations: [] }) };
}

function conversations(env) {
  const a = conversation("a", "all");
  const b = conversation("b");
  env.app.state.conversations = [a, b];
  env.app.selectConversation("a");
  return { a, b };
}

test("clicking the active conversation preserves and saves its draft", () => {
  const env = harness();
  const { a } = conversations(env);
  env.type("An unfinished follow-up");
  env.app.selectConversation("a");
  assert.equal(env.element("question").value, "An unfinished follow-up");
  assert.equal(a.draft, "An unfinished follow-up");
  assert.equal(JSON.parse(env.storage.value).conversations[0].draft, a.draft);
});

test("switching conversations restores each draft and search scope", () => {
  const env = harness();
  const { a, b } = conversations(env);
  env.type("Draft A");
  env.app.selectConversation("b");
  assert.equal(env.element("question").value, "");
  assert.equal(env.app.state.scope, "reference");
  env.type("Draft B");
  env.app.selectConversation("a");
  assert.equal(env.element("question").value, "Draft A");
  assert.equal(env.app.state.scope, "all");
  assert.equal(a.draft, "Draft A");
  assert.equal(b.draft, "Draft B");
});

test("reload restores both conversation drafts and the active composer", () => {
  const env = harness();
  conversations(env);
  env.type("Draft A");
  env.app.selectConversation("b");
  env.type("Draft B");
  const restored = env.reload();
  assert.equal(restored.app.state.activeId, "b");
  assert.equal(restored.element("question").value, "Draft B");
  restored.app.selectConversation("a");
  assert.equal(restored.element("question").value, "Draft A");
});

test("an unsent new conversation survives reload and explicit new chat resets it", () => {
  const env = harness();
  const { a } = conversations(env);
  env.type("Draft A");
  env.app.newConversation();
  assert.equal(a.draft, "Draft A");
  assert.equal(env.element("question").value, "");
  env.type("Unsent new conversation");
  const restored = env.reload();
  assert.equal(restored.element("question").value, "Unsent new conversation");
  restored.app.newConversation();
  assert.equal(restored.element("question").value, "");
  assert.equal(JSON.parse(restored.storage.value).draft, "");
});

test("stored draft values are validated and bounded", () => {
  const storage = { value: JSON.stringify({
    activeId: "a", draft: { invalid: true },
    conversations: [{ ...conversation("a"), draft: "x".repeat(2200) },
      { ...conversation("b"), draft: ["invalid"] }],
  }) };
  const env = harness(storage);
  env.app.restore();
  env.app.restoreDraft();
  assert.equal(env.element("question").value.length, 2000);
  assert.equal(env.app.state.draft, "");
  assert.equal(env.app.state.conversations[1].draft, "");
});

test("submitting consumes only that draft and background completion preserves new typing", async () => {
  let resolve;
  const env = harness(undefined, () => new Promise((done) => { resolve = done; }));
  const { a, b } = conversations(env);
  env.type("Draft A");
  env.app.selectConversation("b");
  env.type("Question B");
  const pending = env.app.sendQuestion(env.element("question").value);
  assert.equal(b.draft, "");
  assert.equal(env.element("question").value, "");
  assert.equal(a.draft, "Draft A");
  env.app.selectConversation("a");
  env.type("Draft A edited while B is pending");
  resolve(answer());
  await pending;
  assert.equal(env.element("question").value, "Draft A edited while B is pending");
  assert.equal(a.draft, "Draft A edited while B is pending");
  assert.equal(env.app.state.scope, "all");
  assert.deepEqual(b.messages.map((message) => message.role), ["user", "answer"]);
  assert.equal(env.timers.size, 0);
});

test("invalid or blocked submissions do not discard drafts", async () => {
  const env = harness();
  const { a } = conversations(env);
  env.type("Keep this draft");
  await env.app.sendQuestion("");
  await env.app.sendQuestion("x".repeat(2001));
  env.app.state.pending = { conversationId: "b" };
  await env.app.sendQuestion("A blocked question");
  assert.equal(a.draft, "Keep this draft");
  assert.equal(env.element("question").value, "Keep this draft");
});

test("the first submission consumes the scratch draft and later typing remains saved", async () => {
  let resolve;
  const env = harness(undefined, () => new Promise((done) => { resolve = done; }));
  env.type("First question");
  const pending = env.app.sendQuestion(env.element("question").value);
  const sent = env.app.state.conversations[0];
  assert.equal(env.app.state.draft, "");
  assert.equal(sent.draft, "");
  assert.equal(env.element("question").value, "");
  env.type("A follow-up typed while waiting");
  resolve(answer());
  await pending;
  assert.equal(sent.draft, "A follow-up typed while waiting");
  const restored = env.reload();
  assert.equal(restored.element("question").value, "A follow-up typed while waiting");
});

test("retrying an earlier error preserves the unsent follow-up and message order", async () => {
  const env = harness();
  const { a } = conversations(env);
  a.messages = [{ id: "u", role: "user", content: "Earlier question" },
    { id: "e", role: "error", content: "Temporary failure", question: "Earlier question", scope: "all" }];
  env.type("Unsent follow-up");
  await env.app.sendQuestion("Earlier question", "all", "e");
  assert.equal(env.element("question").value, "Unsent follow-up");
  assert.equal(a.draft, "Unsent follow-up");
  assert.deepEqual(a.messages.map((message) => message.role), ["user", "answer"]);
});

test("interrupted pending restoration preserves a separately drafted question", async () => {
  let resolve;
  const env = harness(undefined, () => new Promise((done) => { resolve = done; }));
  const { a, b } = conversations(env);
  env.type("Question A");
  const pending = env.app.sendQuestion(env.element("question").value, "all");
  env.app.selectConversation("b");
  env.type("Draft B during search");
  const restored = env.reload();
  assert.equal(restored.element("question").value, "Draft B during search");
  assert.equal(restored.app.state.scope, "reference");
  const interrupted = restored.app.state.conversations.find((item) => item.id === a.id);
  assert.equal(interrupted.messages[1].role, "error");
  assert.equal(interrupted.messages[1].scope, "all");
  assert.equal(restored.app.state.conversations.find((item) => item.id === b.id).draft, "Draft B during search");
  resolve(answer());
  await pending;
});
