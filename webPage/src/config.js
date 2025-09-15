export const API_HTTP_BASE = 'http://158.42.185.235:1337/api';
export const apiUrl = (p) => `${API_HTTP_BASE}${p.startsWith('/') ? '' : '/'}${p}`;
export const WS_BASE = 'ws://158.42.185.235:1337/api';

