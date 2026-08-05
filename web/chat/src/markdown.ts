import DOMPurify from 'dompurify'
import { marked } from 'marked'

const escapeHtml = (text: string) => text.replace(/[&<>"']/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[character]!)
marked.use({
  renderer: { html: ({ text }) => escapeHtml(text) },
})
marked.setOptions({ gfm: true, breaks: true })
export function renderMarkdown(source: string): string {
  const html = marked.parse(source, { async: false }) as string
  return DOMPurify.sanitize(html, {
    USE_PROFILES: { html: true },
    FORBID_TAGS: ['style', 'iframe', 'form', 'input', 'button'],
    FORBID_ATTR: ['style'],
    ALLOWED_URI_REGEXP: /^(?:(?:https?|mailto):|[^a-z]|[a-z+.-]+(?:[^a-z+.-:]|$))/i,
  })
}
