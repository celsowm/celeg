import { useEffect, useMemo, useRef, useState } from 'preact/hooks'
import { renderMarkdown } from './markdown'
import { httpError, parseSSE } from './sse'
import { conversationLimit, load, save, type Store } from './storage'
import type { Conversation, Message, Model, Settings, ToolCall, Usage } from './types'

const uid = () => crypto.randomUUID()
const fresh = (): Conversation => ({ id: uid(), title: 'New conversation', created: Date.now(), updated: Date.now(), messages: [] })
type Chunk = {
  choices?: { delta?: { content?: string; tool_calls?: ToolCall[] }; finish_reason?: string | null }[]
  usage?: Usage
  error?: { message?: string }
}

function Markdown({ text }: { text: string }) {
  const html = useMemo(() => renderMarkdown(text), [text])
  useEffect(() => {
    document.querySelectorAll<HTMLAnchorElement>('.markdown a').forEach(a => { a.target = '_blank'; a.rel = 'noopener noreferrer' })
  }, [html])
  return <div class="markdown" dangerouslySetInnerHTML={{ __html: html }} />
}

function SettingNumber({ label, value, update, step, min, max }: { label: string; value?: number; update: (n?: number) => void; step?: number; min?: number; max?: number }) {
  return <label>{label}<input type="number" value={value ?? ''} step={step} min={min} max={max} placeholder="API default" onInput={e => {
    const raw = e.currentTarget.value
    const number = raw === '' ? undefined : Number(raw)
    update(number === undefined || Number.isFinite(number) ? number : undefined)
  }} /></label>
}

export function App() {
  const initial = useRef(load()).current
  const [conversations, setConversations] = useState(initial.conversations)
  const [activeId, setActiveId] = useState(initial.active)
  const [settings, setSettings] = useState(initial.settings)
  const [models, setModels] = useState<Model[]>([])
  const [draft, setDraft] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [contextNotice, setContextNotice] = useState('')
  const [sidebar, setSidebar] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const [saved, setSaved] = useState(true)
  const controller = useRef<AbortController>()
  const viewport = useRef<HTMLDivElement>(null)
  const composing = useRef(false)
  const active = conversations.find(c => c.id === activeId)

  useEffect(() => { setSaved(save({ conversations, active: activeId, settings })) }, [conversations, activeId, settings])
  useEffect(() => {
    const media = matchMedia('(prefers-color-scheme: dark)')
    const apply = () => { document.documentElement.dataset.theme = settings.theme === 'system' ? (media.matches ? 'dark' : 'light') : settings.theme }
    apply(); media.addEventListener('change', apply); return () => media.removeEventListener('change', apply)
  }, [settings.theme])
  useEffect(() => {
    fetch('/v1/models').then(async r => { if (!r.ok) throw await httpError(r); return r.json() }).then((v: { data?: Model[] } | Model[]) => {
      const list = Array.isArray(v) ? v : v.data || []; setModels(list)
      if (!settings.model && list[0]) setSettings(s => ({ ...s, model: list[0].id }))
    }).catch(e => setError(`Could not load models: ${e instanceof Error ? e.message : e}`))
  }, [])

  const mutate = (id: string, fn: (c: Conversation) => Conversation) => setConversations(cs => cs.map(c => c.id === id ? fn(c) : c))
  const newChat = () => { const c = fresh(); setConversations(cs => [c, ...cs].slice(0, conversationLimit)); setActiveId(c.id); setSidebar(false); setDraft('') }
  const choose = (id: string) => { if (!busy) { setActiveId(id); setSidebar(false) } }
  const nearBottom = () => { const el = viewport.current; return !!el && el.scrollHeight - el.scrollTop - el.clientHeight < 120 }
  const scroll = (force = false) => requestAnimationFrame(() => { const el = viewport.current; if (el && (force || nearBottom())) el.scrollTop = el.scrollHeight })

  async function generate(base: Message[], id: string) {
    if (!settings.model) { setError('Select a model first.'); return }
    const assistant: Message = { id: uid(), role: 'assistant', content: '' }
    mutate(id, c => ({ ...c, updated: Date.now(), messages: [...base, assistant] }))
    setBusy(true); setError(''); setContextNotice(''); scroll(true)
    const abort = new AbortController(); controller.current = abort
    const started = performance.now(); let content = ''; let usage: Usage | undefined; let tools: ToolCall[] = []; let frame = 0
    const flush = () => {
      frame = 0
      mutate(id, c => ({ ...c, messages: c.messages.map(m => m.id === assistant.id ? { ...m, content, usage, toolCalls: tools, elapsed: (performance.now() - started) / 1000 } : m) }))
      scroll()
    }
    let failed = false
    try {
      const optional = Object.fromEntries(Object.entries({ temperature: settings.temperature, top_p: settings.top_p, top_k: settings.top_k, max_tokens: settings.max_tokens, seed: settings.seed }).filter(([, v]) => v !== undefined))
      const messages = [...(settings.systemPrompt.trim() ? [{ role: 'system', content: settings.systemPrompt.trim() }] : []), ...base.map(message => ({
        role: message.role,
        content: message.content || (message.toolCalls?.length ? undefined : ''),
        ...(message.toolCalls?.length ? { tool_calls: message.toolCalls } : {}),
      }))]
      const response = await fetch('/v1/chat/completions', { method: 'POST', headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' }, signal: abort.signal, body: JSON.stringify({ model: settings.model, messages, stream: true, stream_options: { include_usage: true }, ...(settings.enable_thinking === undefined ? {} : { chat_template_kwargs: { enable_thinking: settings.enable_thinking } }), ...optional }) })
      if (!response.ok) throw await httpError(response)
      if (response.headers.get('X-Celeg-Context-Trimmed') === 'true') {
        setContextNotice('Older messages were trimmed to fit the model context window.')
      }
      if (!response.body) throw new Error('Streaming response has no body')
      for await (const event of parseSSE(response.body)) {
        if (event.data === '[DONE]') break
        let chunk: Chunk
        try { chunk = JSON.parse(event.data) } catch { throw new Error('Server sent invalid streaming JSON') }
        if (chunk.error) throw new Error(chunk.error.message || 'Generation failed')
        content += chunk.choices?.[0]?.delta?.content || ''
        const incoming = chunk.choices?.[0]?.delta?.tool_calls
        if (incoming) tools = [...tools, ...incoming]
        if (chunk.usage) usage = chunk.usage
        if (!frame) frame = requestAnimationFrame(flush)
      }
    } catch (e) {
      failed = !abort.signal.aborted
      if (failed) setError(e instanceof Error ? e.message : String(e))
    } finally {
      if (frame) cancelAnimationFrame(frame)
      if (failed || abort.signal.aborted) {
        if (content || tools.length) flush()
        else mutate(id, c => ({ ...c, messages: c.messages.filter(m => m.id !== assistant.id) }))
      } else flush()
      setBusy(false)
      controller.current = undefined
    }
  }

  function send() {
    const text = draft.trim(); if (!text || busy) return
    let c = active
    if (!c) { c = fresh(); setConversations(cs => [c!, ...cs].slice(0, conversationLimit)); setActiveId(c.id) }
    const user: Message = { id: uid(), role: 'user', content: text }
    const base = [...c.messages, user]
    mutate(c.id, old => ({ ...old, title: old.messages.length ? old.title : text.slice(0, 48), messages: base, updated: Date.now() }))
    setDraft(''); void generate(base, c.id)
  }
  function edit(index: number) { if (!active || busy) return; setDraft(active.messages[index].content); mutate(active.id, c => ({ ...c, messages: c.messages.slice(0, index) })) }
  function regenerate() { if (!active || busy) return; const base = active.messages.at(-1)?.role === 'assistant' ? active.messages.slice(0, -1) : active.messages; if (base.length) void generate(base, active.id) }
  function remove(id: string) { if (busy || !confirm('Delete this conversation?')) return; setConversations(cs => cs.filter(c => c.id !== id)); if (id === activeId) setActiveId(undefined) }
  function rename(c: Conversation) { const title = prompt('Conversation name', c.title)?.trim(); if (title) mutate(c.id, old => ({ ...old, title })) }
  function exportData() { const blob = new Blob([JSON.stringify({ exported_at: new Date().toISOString(), conversations }, null, 2)], { type: 'application/json' }); const a = Object.assign(document.createElement('a'), { href: URL.createObjectURL(blob), download: 'celeg-conversations.json' }); a.click(); URL.revokeObjectURL(a.href) }
  const update = <K extends keyof Settings>(key: K, value: Settings[K]) => setSettings(s => ({ ...s, [key]: value }))
  const selected = models.find(m => m.id === settings.model)

  return <div class="shell">
    <button class="scrim" aria-label="Close sidebar" onClick={() => setSidebar(false)} data-open={sidebar} />
    <aside class={sidebar ? 'open' : ''}>
      <div class="brand"><span class="logo">C</span><strong>Celeg</strong><button class="icon mobile" onClick={() => setSidebar(false)} aria-label="Close">×</button></div>
      <button class="new" onClick={newChat} disabled={busy}>＋ New conversation</button>
      <nav aria-label="Conversations">{conversations.map(c => <div key={c.id} class={c.id === activeId ? 'conversation active' : 'conversation'}>
        <button class="title" onClick={() => choose(c.id)} disabled={busy && c.id !== activeId}>{c.title}</button>
        <button class="icon" onClick={() => rename(c)} aria-label={`Rename ${c.title}`}>✎</button><button class="icon" onClick={() => remove(c.id)} aria-label={`Delete ${c.title}`}>×</button>
      </div>)}</nav>
      <div class="sidefoot"><button onClick={exportData} disabled={!conversations.length}>Export JSON</button><button onClick={() => { if (confirm('Clear all local history?')) { setConversations([]); setActiveId(undefined) } }} disabled={busy}>Clear history</button><small>Stored locally · up to {conversationLimit} chats {!saved && '· Storage full—latest changes not saved'}</small></div>
    </aside>
    <main>
      <header><button class="icon mobile" onClick={() => setSidebar(true)} aria-label="Open conversations">☰</button><div><strong>{active?.title || 'New conversation'}</strong>{selected?.celeg?.context_window && <small>{selected.celeg.context_window.toLocaleString()} context</small>}</div><a href="/docs">API docs</a><button onClick={() => setShowSettings(v => !v)} aria-expanded={showSettings}>Settings</button></header>
      {showSettings && <section class="settings" aria-label="Generation settings">
        <label>Model<select value={settings.model} onChange={e => update('model', e.currentTarget.value)}><option value="">Select a model</option>{models.map(m => <option key={m.id} value={m.id}>{m.id}</option>)}</select></label>
        <SettingNumber label="Temperature" value={settings.temperature} update={v => update('temperature', v)} min={0} step={0.1} />
        <SettingNumber label="Top P" value={settings.top_p} update={v => update('top_p', v)} min={0} max={1} step={0.05} />
        <SettingNumber label="Top K" value={settings.top_k} update={v => update('top_k', v)} min={0} step={1} />
        <SettingNumber label="Max tokens" value={settings.max_tokens} update={v => update('max_tokens', v)} min={1} step={1} />
        <SettingNumber label="Seed" value={settings.seed} update={v => update('seed', v)} step={1} />
        <label>Thinking<select value={settings.enable_thinking === undefined ? '' : String(settings.enable_thinking)} onChange={e => update('enable_thinking', e.currentTarget.value === '' ? undefined : e.currentTarget.value === 'true')}><option value="">API default</option><option value="true">Enabled</option><option value="false">Disabled</option></select></label>
        <label>Theme<select value={settings.theme} onChange={e => update('theme', e.currentTarget.value as Settings['theme'])}><option value="system">System</option><option value="light">Light</option><option value="dark">Dark</option></select></label>
        <label class="wide">System prompt<textarea rows={2} value={settings.systemPrompt} onInput={e => update('systemPrompt', e.currentTarget.value)} /></label>
        {selected?.celeg?.capabilities && <small class="wide">Capabilities: {Object.entries(selected.celeg.capabilities).filter(([, supported]) => supported).map(([name]) => name.replaceAll('_', ' ')).join(', ') || 'text chat'}</small>}
      </section>}
      <div class="messages" ref={viewport}>
        {contextNotice && <div class="context-notice" role="status">{contextNotice}</div>}
        {!active?.messages.length && <div class="empty"><span class="logo large">C</span><h1>How can I help?</h1><p>Private, local-first conversations with your Celeg model.</p></div>}
        {active?.messages.map((m, i) => <article class={m.role} key={m.id}><div class="avatar">{m.role === 'user' ? 'You' : 'C'}</div><div class="bubble">
          {m.role === 'assistant' && busy && i === active.messages.length - 1 ? <div class="stream">{m.content || <span class="thinking">Thinking…</span>}</div> : <Markdown text={m.content} />}
          {m.toolCalls?.map((t, toolIndex) => <details class="tool" key={`${m.id}-tool-${toolIndex}`}><summary>Tool call · {t.function?.name || 'unknown'}</summary><pre>{t.function?.arguments || 'No arguments'}</pre><small>Display only — not executed</small></details>)}
          <div class="meta">{m.role === 'user' && <button onClick={() => edit(i)}>Edit</button>}{m.usage && <span>{m.usage.prompt_tokens ?? 0} in · {m.usage.completion_tokens ?? 0} out · {m.usage.total_tokens ?? 0} total</span>}{m.elapsed !== undefined && <span>{m.elapsed.toFixed(1)}s</span>}</div>
        </div></article>)}
        {error && <div class="error" role="alert">{error}<button class="icon" onClick={() => setError('')} aria-label="Dismiss">×</button></div>}
      </div>
      <footer><div class="composer"><textarea aria-label="Message" placeholder="Message Celeg…" rows={1} value={draft} onInput={e => setDraft(e.currentTarget.value)} onCompositionStart={() => composing.current = true} onCompositionEnd={() => composing.current = false} onKeyDown={e => { if (e.key === 'Escape' && busy) controller.current?.abort(); if (e.key === 'Enter' && !e.shiftKey && !composing.current) { e.preventDefault(); send() } }} />{busy ? <button class="send stop" onClick={() => controller.current?.abort()} aria-label="Stop generation">■</button> : <button class="send" onClick={send} disabled={!draft.trim() || !settings.model} aria-label="Send message">↑</button>}</div><div class="under"><span>Enter to send · Shift+Enter for newline</span>{active?.messages.at(-1)?.role === 'assistant' && !busy && <button onClick={regenerate}>↻ Regenerate</button>}</div><div class="status" aria-live="polite">{busy ? 'Generating response. Press Escape to stop.' : error || ''}</div></footer>
    </main>
  </div>
}
