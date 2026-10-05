import { useEffect, useMemo, useRef, useState } from 'react'
import './App.css'
import { askQuestion } from './services/api'

const SUGGESTED_QUESTIONS = [
  'Who is the HOD of Information Technology?',
  'What is the fee structure for 2026-27?',
  'What scholarships are available?',
  'What courses are included in the BE syllabus?',
  'Show me information about the grievance redressal committee.',
]

function createId(prefix) {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return `${prefix}-${crypto.randomUUID()}`
  }
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`
}

function makeEmptyConversation() {
  return {
    id: createId('conv'),
    title: 'New chat',
    createdAt: new Date().toISOString(),
    messages: [],
  }
}

function generateTitleFromQuestion(question) {
  const cleaned = question.trim().replace(/\s+/g, ' ')
  const tokens = cleaned.split(' ')
  const raw = tokens.slice(0, 6).join(' ')
  if (!raw) return 'New chat'
  return raw.length > 28 ? `${raw.slice(0, 25).trim()}...` : raw
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
}

function formatInlineMarkdown(value) {
  let formatted = escapeHtml(value)
  formatted = formatted.replace(/`([^`]+)`/g, '<code>$1</code>')
  formatted = formatted.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
  formatted = formatted.replace(/__(.+?)__/g, '<strong>$1</strong>')
  formatted = formatted.replace(/\*([^*]+)\*/g, '<em>$1</em>')
  formatted = formatted.replace(/_([^_]+)_/g, '<em>$1</em>')
  return formatted
}

function markdownToHtml(markdown) {
  if (!markdown || !markdown.trim()) {
    return ''
  }

  const lines = markdown.replace(/\r\n/g, '\n').split('\n')
  const htmlBlocks = []
  let listItems = []
  let orderedListItems = []
  let paragraph = []

  const flushParagraph = () => {
    if (!paragraph.length) return
    const combined = paragraph.join(' ').trim()
    if (combined) {
      htmlBlocks.push(`<p>${formatInlineMarkdown(combined)}</p>`)
    }
    paragraph = []
  }

  const flushLists = () => {
    if (listItems.length) {
      htmlBlocks.push(`<ul>${listItems.join('')}</ul>`)
      listItems = []
    }

    if (orderedListItems.length) {
      htmlBlocks.push(`<ol>${orderedListItems.join('')}</ol>`)
      orderedListItems = []
    }
  }

  for (const rawLine of lines) {
    const line = rawLine.trim()

    if (!line) {
      flushParagraph()
      flushLists()
      continue
    }

    const headingMatch = line.match(/^(#{1,6})\s+(.*)$/)
    if (headingMatch) {
      flushParagraph()
      flushLists()
      const level = headingMatch[1].length
      const text = formatInlineMarkdown(headingMatch[2])
      htmlBlocks.push(`<h${level}>${text}</h${level}>`)
      continue
    }

    if (/^[-*]\s+/.test(line)) {
      flushParagraph()
      listItems.push(`<li>${formatInlineMarkdown(line.replace(/^[-*]\s+/, ''))}</li>`)
      continue
    }

    if (/^\d+\.\s+/.test(line)) {
      flushParagraph()
      orderedListItems.push(`<li>${formatInlineMarkdown(line.replace(/^\d+\.\s+/, ''))}</li>`)
      continue
    }

    flushLists()
    paragraph.push(line)
  }

  flushParagraph()
  flushLists()

  return htmlBlocks.join('')
}

function groupByDate(conversations) {
  const today = new Date()
  const todayKey = today.toDateString()

  return conversations.reduce(
    (accumulator, conversation) => {
      const createdAt = new Date(conversation.createdAt)
      const bucket = createdAt.toDateString() === todayKey ? 'Today' : 'Previous conversations'
      if (!accumulator[bucket]) {
        accumulator[bucket] = []
      }
      accumulator[bucket].push(conversation)
      return accumulator
    },
    { Today: [], 'Previous conversations': [] },
  )
}

function buildStatusNotice(status) {
  switch (status) {
    case 'SUFFICIENT':
      return null
    case 'INSUFFICIENT_KNOWLEDGE':
      return 'Not enough information was found in the available MHSSCE documents.'
    case 'CONFLICTING_EVIDENCE':
      return 'Some official documents contain conflicting information.'
    default:
      return 'The assistant could not complete the response with the available MHSSCE evidence.'
  }
}

function App() {
  const [conversations, setConversations] = useState(() => {
    const persisted = window.localStorage.getItem('mhssce-chat-history')
    if (!persisted) {
      return [makeEmptyConversation()]
    }

    try {
      const parsed = JSON.parse(persisted)
      return Array.isArray(parsed) && parsed.length ? parsed : [makeEmptyConversation()]
    } catch {
      return [makeEmptyConversation()]
    }
  })

  const [activeConversationId, setActiveConversationId] = useState(() => {
    return window.localStorage.getItem('mhssce-active-chat') || null
  })
  const [draft, setDraft] = useState('')
  const [loading, setLoading] = useState(false)
  const [loadingPhase, setLoadingPhase] = useState('searching')
  const [error, setError] = useState('')
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false)
  const [darkMode, setDarkMode] = useState(() => {
    const stored = window.localStorage.getItem('mhssce-theme')
    return stored ? stored === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches
  })
  const [debugMode, setDebugMode] = useState(() => {
    const stored = window.localStorage.getItem('mhssce-debug-mode')
    return stored === 'true'
  })
  const [expandedSources, setExpandedSources] = useState({})
  const [expandedReasoning, setExpandedReasoning] = useState({})
  const textareaRef = useRef(null)

  const activeConversation =
    conversations.find((conversation) => conversation.id === activeConversationId) || conversations[0]

  useEffect(() => {
    if (!conversations.length) {
      const fallback = makeEmptyConversation()
      setConversations([fallback])
      setActiveConversationId(fallback.id)
      return
    }

    if (!activeConversationId || !conversations.some((conversation) => conversation.id === activeConversationId)) {
      setActiveConversationId(conversations[0].id)
    }
  }, [activeConversationId, conversations])

  useEffect(() => {
    window.localStorage.setItem('mhssce-chat-history', JSON.stringify(conversations))
    if (activeConversationId) {
      window.localStorage.setItem('mhssce-active-chat', activeConversationId)
    }
  }, [activeConversationId, conversations])

  useEffect(() => {
    window.localStorage.setItem('mhssce-theme', darkMode ? 'dark' : 'light')
    document.documentElement.setAttribute('data-theme', darkMode ? 'dark' : 'light')
  }, [darkMode])

  useEffect(() => {
    window.localStorage.setItem('mhssce-debug-mode', String(debugMode))
  }, [debugMode])

  useEffect(() => {
    if (!textareaRef.current) return
    textareaRef.current.style.height = 'auto'
    textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 180)}px`
  }, [draft])

  useEffect(() => {
    if (!loading) return
    const timeoutId = setTimeout(() => {
      setLoadingPhase('generating')
    }, 900)
    return () => clearTimeout(timeoutId)
  }, [loading])

  useEffect(() => {
    const handleKeyToggle = (event) => {
      if (event.ctrlKey && event.altKey && event.key.toLowerCase() === 'd') {
        event.preventDefault()
        setDebugMode((current) => !current)
      }
    }

    window.addEventListener('keydown', handleKeyToggle)
    return () => window.removeEventListener('keydown', handleKeyToggle)
  }, [])

  const groupedConversations = useMemo(() => groupByDate(conversations), [conversations])

  const updateConversation = (conversationId, updater) => {
    setConversations((current) =>
      current.map((conversation) =>
        conversation.id === conversationId ? updater(conversation) : conversation,
      ),
    )
  }

  const createNewChat = () => {
    const nextConversation = makeEmptyConversation()
    setConversations((current) => [nextConversation, ...current])
    setActiveConversationId(nextConversation.id)
    setDraft('')
    setError('')
    setLoading(false)
    setMobileSidebarOpen(false)
  }

  const renameConversation = (conversationId) => {
    const current = conversations.find((conversation) => conversation.id === conversationId)
    if (!current) return

    const nextTitle = window.prompt('Rename conversation', current.title)
    if (!nextTitle || !nextTitle.trim()) return

    updateConversation(conversationId, (conversation) => ({
      ...conversation,
      title: nextTitle.trim(),
    }))
  }

  const deleteConversation = (conversationId) => {
    if (conversations.length === 1) {
      const replacement = makeEmptyConversation()
      setConversations([replacement])
      setActiveConversationId(replacement.id)
      return
    }

    const nextConversations = conversations.filter((conversation) => conversation.id !== conversationId)
    setConversations(nextConversations)
    setActiveConversationId(nextConversations[0].id)
  }

  const sendQuestion = async (questionOverride) => {
    const question = ((questionOverride ?? draft) || '').trim()
    if (!question || loading) return

    const userMessage = {
      id: createId('user'),
      role: 'user',
      content: question,
      timestamp: new Date().toISOString(),
    }

    const loadingMessageId = createId('assistant')
    const loadingMessage = {
      id: loadingMessageId,
      role: 'assistant',
      status: 'loading',
      loadingStep: loadingPhase,
      content: '',
      sources: [],
      timestamp: new Date().toISOString(),
    }

    setDraft('')
    setError('')
    setLoading(true)
    setLoadingPhase('searching')

    updateConversation(activeConversation.id, (conversation) => ({
      ...conversation,
      title:
        conversation.messages.length === 0
          ? generateTitleFromQuestion(question)
          : conversation.title,
      messages: [...conversation.messages, userMessage, loadingMessage],
    }))

    try {
      const response = await askQuestion(question, debugMode)

      const assistantMessage = {
        id: createId('assistant-reply'),
        role: 'assistant',
        status: response.status || 'UNKNOWN',
        answer: response.answer || 'No answer was returned by the backend.',
        sources: response.sources || [],
        conflicting_evidence: Boolean(response.conflicting_evidence),
        requires_personal_context: Boolean(response.requires_personal_context),
        confidence_note: response.confidence_note || '',
        generation: response.generation || 'abstention',
        timestamp: new Date().toISOString(),
      }

      updateConversation(activeConversation.id, (conversation) => ({
        ...conversation,
        title:
          conversation.messages.length <= 1
            ? generateTitleFromQuestion(question)
            : conversation.title,
        messages: conversation.messages.map((message) =>
          message.id === loadingMessageId ? assistantMessage : message,
        ),
      }))
    } catch (caughtError) {
      const assistantMessage = {
        id: createId('assistant-error'),
        role: 'assistant',
        status: 'ERROR',
        answer: caughtError?.message || 'I could not connect to the MHSSCE knowledge service.',
        sources: [],
        conflicting_evidence: false,
        requires_personal_context: false,
        confidence_note: '',
        generation: 'abstention',
        timestamp: new Date().toISOString(),
      }

      updateConversation(activeConversation.id, (conversation) => ({
        ...conversation,
        messages: conversation.messages.map((message) =>
          message.id === loadingMessageId ? assistantMessage : message,
        ),
      }))
      setError(caughtError?.message || 'The request could not be completed.')
    } finally {
      setLoading(false)
      setLoadingPhase('searching')
    }
  }

  const handleTextareaKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      sendQuestion()
    }
  }

  const currentMessages = activeConversation?.messages || []

  return (
    <div className="app-shell" style={{ '--sidebar-width': sidebarOpen ? '280px' : '92px' }}>
      <aside className={`sidebar ${sidebarOpen ? '' : 'collapsed'} ${mobileSidebarOpen ? 'mobile-open' : ''}`}>
        <div className="sidebar-top">
          <button type="button" className="new-chat-button" onClick={createNewChat}>
            + New Chat
          </button>
          <button
            type="button"
            className="sidebar-toggle desktop-toggle"
            onClick={() => setSidebarOpen((current) => !current)}
            aria-label="Toggle sidebar"
          >
            {sidebarOpen ? '⟨' : '⟩'}
          </button>
        </div>

        <div className="sidebar-brand">
          <div className="brand-badge">MH</div>
          <div className="brand-copy">
            <strong>MHSSCE Knowledge Assistant</strong>
            <span>College information powered by RAG</span>
          </div>
        </div>

        <div className="history-groups">
          {Object.entries(groupedConversations).map(([label, items]) => {
            if (!items.length) return null

            return (
              <div className="history-group" key={label}>
                <h3>{label}</h3>
                <ul>
                  {items.map((conversation) => (
                    <li
                      key={conversation.id}
                      className={conversation.id === activeConversation.id ? 'active' : ''}
                    >
                      <button
                        type="button"
                        className="conversation-item"
                        onClick={() => {
                          setActiveConversationId(conversation.id)
                          setMobileSidebarOpen(false)
                        }}
                      >
                        <span>{conversation.title || 'New chat'}</span>
                        <small>
                          {new Date(conversation.createdAt).toLocaleTimeString([], {
                            hour: 'numeric',
                            minute: '2-digit',
                          })}
                        </small>
                      </button>

                      <div className="conversation-actions">
                        <button
                          type="button"
                          aria-label="Rename conversation"
                          onClick={(event) => {
                            event.stopPropagation()
                            renameConversation(conversation.id)
                          }}
                        >
                          Rename
                        </button>
                        <button
                          type="button"
                          aria-label="Delete conversation"
                          onClick={(event) => {
                            event.stopPropagation()
                            deleteConversation(conversation.id)
                          }}
                        >
                          Delete
                        </button>
                      </div>
                    </li>
                  ))}
                </ul>
              </div>
            )
          })}
        </div>
      </aside>

      <main className="chat-panel">
        <header className="chat-header">
          <div className="header-main">
            <button
              type="button"
              className="sidebar-toggle mobile-toggle"
              onClick={() => setMobileSidebarOpen((current) => !current)}
              aria-label="Open sidebar"
            >
              ☰
            </button>
            <div>
              <h1>MHSSCE Knowledge Assistant</h1>
              <div className="header-meta">
                <span>RAG-powered • Official College Documents</span>
                <span className="online-indicator">● Online</span>
              </div>
            </div>
          </div>

          <div className="header-actions">
            <button
              type="button"
              className="theme-toggle"
              onClick={() => setDarkMode((current) => !current)}
            >
              {darkMode ? '☀️ Light' : '🌙 Dark'}
            </button>
            {debugMode && (
              <span className="debug-pill">Debug</span>
            )}
          </div>
        </header>

        <div className="chat-body">
          {currentMessages.length === 0 ? (
            <div className="welcome-screen">
              <div className="welcome-card">
                <p className="eyebrow">MHSSCE Knowledge Assistant</p>
                <h2>How can I help you with MHSSCE?</h2>
                <p>
                  Ask questions about college departments, fees, scholarships, syllabus,
                  academic information, and official documents.
                </p>

                <div className="suggestions">
                  {SUGGESTED_QUESTIONS.map((question) => (
                    <button
                      key={question}
                      type="button"
                      className="suggestion-button"
                      onClick={() => sendQuestion(question)}
                    >
                      {question}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="messages">
              {currentMessages.map((message) => {
                const isAssistant = message.role === 'assistant'
                const sourceKey = `sources-${message.id}`
                const reasoningKey = `reasoning-${message.id}`

                return (
                  <div key={message.id} className={`message-row ${isAssistant ? 'assistant-row' : 'user-row'}`}>
                    <div className={`message-bubble ${isAssistant ? 'assistant-message' : 'user-message'}`}>
                      {isAssistant ? (
                        <>
                          {message.status === 'loading' ? (
                            <div className="typing-indicator-wrap">
                              <span className="typing-label">
                                {message.loadingStep === 'generating'
                                  ? 'Generating grounded answer...'
                                  : 'Searching official MHSSCE documents...'}
                              </span>
                              <span className="typing-dots" aria-hidden="true">
                                <span />
                                <span />
                                <span />
                              </span>
                            </div>
                          ) : (
                            <>
                              {message.status === 'ERROR' && <div className="notice error-notice">{message.answer}</div>}
                              {message.status === 'INSUFFICIENT_KNOWLEDGE' && (
                                <div className="notice info-notice">
                                  {buildStatusNotice(message.status)}
                                </div>
                              )}
                              {message.status === 'CONFLICTING_EVIDENCE' && (
                                <div className="notice warning-notice">
                                  {buildStatusNotice(message.status)}
                                </div>
                              )}

                              <div
                                className="message-content"
                                dangerouslySetInnerHTML={{ __html: markdownToHtml(message.answer || '') }}
                              />

                              {message.confidence_note && (
                                <div className="confidence-note">{message.confidence_note}</div>
                              )}

                              {message.sources && message.sources.length > 0 && (
                                <div className="source-section">
                                  <h3>Sources</h3>
                                  <div className="source-grid">
                                    {message.sources.map((source, index) => {
                                      const isExpanded = expandedSources[sourceKey + index]
                                      return (
                                        <button
                                          key={`${source.document}-${source.page || 'n/a'}-${index}`}
                                          type="button"
                                          className={`source-card ${isExpanded ? 'expanded' : ''}`}
                                          onClick={() =>
                                            setExpandedSources((current) => ({
                                              ...current,
                                              [sourceKey + index]: !isExpanded,
                                            }))
                                          }
                                        >
                                          <span className="source-icon">📄</span>
                                          <div className="source-content">
                                            <strong>{source.document || 'Official document'}</strong>
                                            {source.page ? <span>Page {source.page}</span> : <span>Page not provided</span>}
                                            {source.academic_year && <span>Academic Year: {source.academic_year}</span>}
                                            {source.document_type && <span>Type: {source.document_type}</span>}
                                            {isExpanded && source.chunk_id && <span>Chunk ID: {source.chunk_id}</span>}
                                            {debugMode && source.retrieval_score != null && (
                                              <span>Score: {Number(source.retrieval_score).toFixed(4)}</span>
                                            )}
                                          </div>
                                        </button>
                                      )
                                    })}
                                  </div>
                                </div>
                              )}

                              <div className="rationale-section">
                                <button
                                  type="button"
                                  className="rationale-toggle"
                                  onClick={() =>
                                    setExpandedReasoning((current) => ({
                                      ...current,
                                      [reasoningKey]: !current[reasoningKey],
                                    }))
                                  }
                                >
                                  How was this answer generated? {expandedReasoning[reasoningKey] ? '▴' : '▾'}
                                </button>

                                {expandedReasoning[reasoningKey] && (
                                  <div className="rationale-panel">
                                    <div className="rationale-pipeline">
                                      <span>Question</span>
                                      <span>↓</span>
                                      <span>Query understanding</span>
                                      <span>↓</span>
                                      <span>Document retrieval</span>
                                      <span>↓</span>
                                      <span>Evidence validation</span>
                                      <span>↓</span>
                                      <span>Gemini</span>
                                      <span>↓</span>
                                      <span>Answer + citations</span>
                                    </div>

                                    <p>
                                      Your question is first searched against the college's indexed
                                      documents. The system selects relevant passages, checks whether there is
                                      enough evidence, and then generates an answer grounded in those sources.
                                    </p>

                                    <div className="evidence-list">
                                      <h4>Evidence used</h4>
                                      {message.sources && message.sources.length > 0 ? (
                                        <ol>
                                          {message.sources.map((source, index) => (
                                            <li key={`${source.document}-${index}`}>
                                              {source.document}
                                              {source.page ? ` — Page ${source.page}` : ''}
                                              {source.academic_year ? ` — Academic Year: ${source.academic_year}` : ''}
                                            </li>
                                          ))}
                                        </ol>
                                      ) : (
                                        <p>No document sources were returned for this answer.</p>
                                      )}
                                    </div>

                                    <p className="rationale-summary">
                                      This answer was generated from official MHSSCE documents indexed in the
                                      college knowledge base.
                                    </p>
                                  </div>
                                )}
                              </div>
                            </>
                          )}
                        </>
                      ) : (
                        <div className="message-content user-content">
                          {message.content}
                        </div>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>

        <div className="composer-panel">
          {error && <div className="global-error">{error}</div>}

          <div className="composer-box">
            <textarea
              ref={textareaRef}
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={handleTextareaKeyDown}
              placeholder="Ask anything about MHSSCE..."
              rows={1}
              disabled={loading}
            />
            <button
              type="button"
              className="send-button"
              onClick={() => sendQuestion()}
              disabled={!draft.trim() || loading}
            >
              {loading ? 'Sending...' : 'Send'}
            </button>
          </div>
        </div>
      </main>
    </div>
  )
}

export default App
