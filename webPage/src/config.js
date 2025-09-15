// config.js — always use same-origin unless explicitly overridden

// API path prefix (default '/api')
const RAW_BASE = process.env.REACT_APP_API_BASE_PATH || '/api';
const API_BASE_PATH = `/${RAW_BASE.replace(/^\/|\/$/g, '')}`; // normalize: '/api'

// Same-origin host/port/protocol (matches the page origin)
const ORIGIN_HTTP = window.location.origin;
const ORIGIN_WS = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}`;

// Allow full override via a single env var if you ever need it
// export const API_HTTP_BASE = process.env.REACT_APP_API_BASE || `${ORIGIN_HTTP}${API_BASE_PATH}`;
export const API_HTTP_BASE = 'http://158.42.185.235:1337/api';
export const WS_BASE = process.env.REACT_APP_WS_BASE || `${ORIGIN_WS}${API_BASE_PATH}`;

// Helper to build URLs
export const apiUrl = (path) => `${API_HTTP_BASE}${path.startsWith('/') ? '' : '/'}${path}`;






