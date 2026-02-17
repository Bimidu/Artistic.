import React, { useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { setToken } from '@utils/storage';
import { useAuth } from '@hooks/useAuth';

export const GoogleCallbackPage = () => {
    const navigate = useNavigate();
    const [searchParams] = useSearchParams();
    const { refreshUser } = useAuth();

    useEffect(() => {
        const token = searchParams.get('token');
        const error = searchParams.get('error');

        if (error) {
            console.error('Google OAuth error:', error);
            navigate('/login?error=google_oauth_failed');
            return;
        }

        if (token) {
            // Save token and refresh user data
            setToken(token);

            // Refresh user data and redirect to home
            refreshUser()
                .then(() => {
                    navigate('/');
                })
                .catch((err) => {
                    console.error('Failed to refresh user:', err);
                    navigate('/login');
                });
        } else {
            // No token, redirect to login
            navigate('/login');
        }
    }, [searchParams, navigate, refreshUser]);

    return (
        <div className="min-h-screen flex items-center justify-center bg-gradient-to-b from-primary-100 to-white">
            <div className="text-center">
                <div className="spinner w-12 h-12 mx-auto mb-4 text-primary-900"></div>
                <p className="text-primary-700 text-lg">Completing Google sign-in...</p>
                <p className="text-primary-500 text-sm mt-2">Please wait</p>
            </div>
        </div>
    );
};
