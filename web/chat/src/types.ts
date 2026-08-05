export type ToolCall = { id?: string; type?: string; function?: { name?: string; arguments?: string } }
export type Message = { id: string; role: 'user' | 'assistant'; content: string; toolCalls?: ToolCall[]; usage?: Usage; elapsed?: number }
export type Usage = { prompt_tokens?: number; completion_tokens?: number; total_tokens?: number }
export type Conversation = { id: string; title: string; created: number; updated: number; messages: Message[] }
export type Settings = { model: string; temperature?: number; top_p?: number; top_k?: number; max_tokens?: number; seed?: number; enable_thinking?: boolean; systemPrompt: string; theme: 'system' | 'light' | 'dark' }
export type ModelCapabilities = { vision?: boolean; tool_calls?: boolean; parallel_tool_calls?: boolean; thinking?: boolean }
export type Model = { id: string; celeg?: { context_window?: number; capabilities?: ModelCapabilities } }
