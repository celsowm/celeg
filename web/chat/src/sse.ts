export type SSEEvent = { data: string; event?: string }

export async function* parseSSE(stream: ReadableStream<Uint8Array>): AsyncGenerator<SSEEvent> {
  const reader = stream.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let data: string[] = []
  let event: string | undefined
  const line = (raw: string): SSEEvent | undefined => {
    const value = raw.endsWith('\r') ? raw.slice(0, -1) : raw
    if (value === '') {
      if (!data.length) { event = undefined; return }
      const result = { data: data.join('\n'), event }
      data = []; event = undefined
      return result
    }
    if (value.startsWith(':')) return
    const colon = value.indexOf(':')
    const field = colon < 0 ? value : value.slice(0, colon)
    let content = colon < 0 ? '' : value.slice(colon + 1)
    if (content.startsWith(' ')) content = content.slice(1)
    if (field === 'data') data.push(content)
    else if (field === 'event') event = content
  }
  try {
    while (true) {
      const { value, done } = await reader.read()
      buffer += decoder.decode(value, { stream: !done })
      let end: number
      while ((end = buffer.indexOf('\n')) >= 0) {
        const result = line(buffer.slice(0, end)); buffer = buffer.slice(end + 1)
        if (result) yield result
      }
      if (done) break
    }
    if (buffer) { const result = line(buffer); if (result) yield result }
    const result = line(''); if (result) yield result
  } finally { reader.releaseLock() }
}

export async function httpError(response: Response): Promise<Error> {
  let message = `${response.status} ${response.statusText}`
  try {
    const body = await response.json() as { error?: string | { message?: string }; message?: string }
    message = typeof body.error === 'string' ? body.error : body.error?.message || body.message || message
  } catch { /* non-JSON response */ }
  return new Error(message)
}
