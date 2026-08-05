// @vitest-environment jsdom
import { describe, expect, it } from 'vitest'
import { renderMarkdown } from './markdown'
describe('markdown', () => {
  it('removes raw executable HTML and unsafe links', () => {
    const html = renderMarkdown('<img src=x onerror=alert(1)> [bad](javascript:alert(1)) <script>alert(1)</script>')
    expect(html).not.toMatch(/<img|href=|<script/i)
    expect(renderMarkdown('<b>raw</b>')).toContain('&lt;b&gt;raw&lt;/b&gt;')
  })
})
