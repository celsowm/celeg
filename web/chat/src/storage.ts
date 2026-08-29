import type { Conversation, Settings } from './types'

const KEY = 'celeg-chat-v1'
const LIMIT = 40
export type Store = { conversations: Conversation[]; active?: string; settings: Settings }
export const defaults: Settings = { model: '', systemPrompt: '', theme: 'system', enable_thinking: false }
export function load(): Store {
  try {
    const value = JSON.parse(localStorage.getItem(KEY) || '') as Partial<Store>
    return { conversations: Array.isArray(value.conversations) ? value.conversations.slice(0, LIMIT) : [], active: value.active, settings: { ...defaults, ...value.settings } }
  } catch { return { conversations: [], settings: defaults } }
}
export function save(store: Store): boolean {
  try { localStorage.setItem(KEY, JSON.stringify({ ...store, conversations: store.conversations.slice(0, LIMIT) })); return true }
  catch { return false }
}
export const conversationLimit = LIMIT
