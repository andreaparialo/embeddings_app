// ================================================
// API Client Utility
// ================================================

export class ApiClient {
    constructor(baseUrl = '') {
        this.baseUrl = baseUrl;
    }

    async get(endpoint, params = {}) {
        const url = new URL(endpoint, window.location.origin);
        Object.keys(params).forEach(key => {
            if (params[key] !== undefined && params[key] !== null) {
                url.searchParams.append(key, params[key]);
            }
        });

        const response = await fetch(url, {
            method: 'GET',
            headers: {
                'Accept': 'application/json'
            }
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    }

    async post(endpoint, data) {
        const response = await fetch(endpoint, {
            method: 'POST',
            body: data
        });

        if (!response.ok) {
            // Try to parse error response
            try {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            } catch {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
        }

        return await response.json();
    }

    async postJSON(endpoint, data) {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            // Try to parse error response
            try {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            } catch {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
        }

        return await response.json();
    }

    async postBlob(endpoint, data) {
        const response = await fetch(endpoint, {
            method: 'POST',
            body: data
        });

        if (!response.ok) {
            // Try to parse error response
            try {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            } catch {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
        }

        return await response.blob();
    }

    async delete(endpoint) {
        const response = await fetch(endpoint, {
            method: 'DELETE',
            headers: {
                'Accept': 'application/json'
            }
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    }
} 