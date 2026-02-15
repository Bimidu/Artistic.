import api from './api';

export const authService = {
    async register(name, email, password) {
        const response = await api.post('/auth/register', {
            full_name: name,
            email,
            password,
        });
        return response.data;
    },

    async login(email, password) {
        const response = await api.post('/auth/login', {
            email,
            password,
        });
        return response.data;
    },

    async logout() {
        await api.post('/auth/logout');
    },

    async getCurrentUser() {
        const response = await api.get('/auth/me');
        return response.data;
    },

    async requestPasswordReset(email) {
        const response = await api.post('/auth/password-reset-request', {
            email,
        });
        return response.data;
    },

    async confirmPasswordReset(token, newPassword) {
        const response = await api.post('/auth/password-reset-confirm', {
            token,
            new_password: newPassword,
        });
        return response.data;
    },

    getGoogleAuthUrl() {
        return `${api.defaults.baseURL}/auth/google/login`;
    },
};
