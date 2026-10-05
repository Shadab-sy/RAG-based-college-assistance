const DEFAULT_BASE_URL = 'http://127.0.0.1:8000';

function resolveBaseUrl() {
  const configured = import.meta.env.VITE_API_URL;
  if (configured && configured.trim()) {
    return configured.replace(/\/$/, '');
  }
  return DEFAULT_BASE_URL;
}

const API_BASE_URL = resolveBaseUrl();

export async function askQuestion(question, includeScores = false) {
  const trimmed = String(question || '').trim();

  if (!trimmed) {
    throw new Error('Please enter a question before sending.');
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 25000);

  try {
    const response = await fetch(`${API_BASE_URL}/api/ask?include_scores=${includeScores}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ question: trimmed }),
      signal: controller.signal,
    });

    const text = await response.text();
    let payload = null;

    if (text) {
      try {
        payload = JSON.parse(text);
      } catch {
        payload = null;
      }
    }

    if (!response.ok) {
      const detail = payload?.detail || payload?.message || 'The MHSSCE knowledge service is unavailable right now.';
      throw new Error(detail);
    }

    if (!payload || typeof payload.answer !== 'string') {
      throw new Error('The backend returned an invalid response.');
    }

    return payload;
  } catch (error) {
    if (error.name === 'AbortError') {
      throw new Error('The request timed out. Please try again.');
    }

    if (error instanceof Error) {
      throw error;
    }

    throw new Error('An unexpected error occurred while contacting the backend.');
  } finally {
    clearTimeout(timeoutId);
  }
}
