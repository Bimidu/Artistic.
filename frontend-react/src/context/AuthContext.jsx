import React, { createContext, useState, useEffect, useCallback } from 'react';
import { authService } from '@services/authService';
import { getToken, setToken, removeToken } from '@utils/storage';

export const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);
    const [isAuthenticated, setIsAuthenticated] = useState(false);

    const loadUser = useCallback(async () => {
        const token = getToken();
        if (!token) {
            setLoading(false);
            return;
        }

        try {
            const userData = await authService.getCurrentUser();
            setUser(userData);
            setIsAuthenticated(true);
        } catch (error) {
            console.error('Failed to load user:', error);
            removeToken();
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        loadUser();
    }, [loadUser]);

    const login = async (email, password) => {
        const data = await authService.login(email, password);
        setToken(data.access_token);
        setUser(data.user);
        setIsAuthenticated(true);
        return data;
    };

    const register = async (name, email, password, role = 'user') => {
        const data = await authService.register(name, email, password, role);
        setToken(data.access_token);
        setUser(data.user);
        setIsAuthenticated(true);
        return data;
    };

    const logout = async () => {
        try {
            await authService.logout();
        } catch (error) {
            console.error('Logout error:', error);
        } finally {
            removeToken();
            setUser(null);
            setIsAuthenticated(false);
        }
    };

    const value = {
        user,
        isAuthenticated,
        loading,
        login,
        register,
        logout,
        refreshUser: loadUser,
    };

    return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};
