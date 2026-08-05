import { describe, expect, it } from 'vitest'
import { parseSSE } from './sse'

function chunks(parts: string[]) { return new ReadableStream({ start(c) { parts.forEach(p => c.enqueue(new TextEncoder().encode(p))); c.close() } }) }
describe('parseSSE', () => {
  it('handles fragmentation, CRLF, multiple and multiline events', async () => {
    const result = []
    for await (const event of parseSSE(chunks(['data: {"a"', ':1}\r\n\r', '\ndata: first\ndata: second\n\ndata: [DONE]\n\n']))) result.push(event.data)
    expect(result).toEqual(['{"a":1}', 'first\nsecond', '[DONE]'])
  })
})
