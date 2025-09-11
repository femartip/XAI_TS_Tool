// Centralized API configuration for HTTP and WebSocket endpoints
// Prefer environment variables; fall back to window location for same-host deployments

const API_HOST = process.env.REACT_APP_API_HOST || window.location.hostname;
const API_PORT = process.env.REACT_APP_API_PORT || '8000';

const HTTP_PROTOCOL = (process.env.REACT_APP_API_PROTOCOL || window.location.protocol.replace(':', '')) === 'https' ? 'https' : 'http';
const WS_PROTOCOL = HTTP_PROTOCOL === 'https' ? 'wss' : 'ws';

export const API_HTTP_BASE = `${HTTP_PROTOCOL}://${API_HOST}:${API_PORT}`;
export const WS_BASE = `${WS_PROTOCOL}://${API_HOST}:${API_PORT}`;

// Helper to build full URLs
export const apiUrl = (path) => `${API_HTTP_BASE}${path.startsWith('/') ? '' : '/'}${path}`;

