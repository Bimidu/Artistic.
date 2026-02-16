import React from 'react';
import { createBrowserRouter, Navigate } from 'react-router-dom';
import { ProtectedRoute } from '@components/auth/ProtectedRoute';

// Pages
import { LoginPage } from '@pages/Auth/LoginPage';
import { RegisterPage } from '@pages/Auth/RegisterPage';
import { GoogleCallbackPage } from '@pages/Auth/GoogleCallbackPage';
import { HomePage } from '@pages/Home/HomePage';
import { PredictionPage } from '@pages/UserMode/PredictionPage';
import { TrainingPage } from '@pages/Training/TrainingPage';
import { ReportsPage } from '@pages/Reports/ReportsPage';

export const router = createBrowserRouter([
    {
        path: '/login',
        element: <LoginPage />,
    },
    {
        path: '/register',
        element: <RegisterPage />,
    },
    {
        path: '/auth/google/callback',
        element: <GoogleCallbackPage />,
    },
    {
        path: '/',
        element: (
            <ProtectedRoute>
                <HomePage />
            </ProtectedRoute>
        ),
    },
    {
        path: '/predict',
        element: (
            <ProtectedRoute>
                <PredictionPage />
            </ProtectedRoute>
        ),
    },
    {
        path: '/training',
        element: (
            <ProtectedRoute>
                <TrainingPage />
            </ProtectedRoute>
        ),
    },
    {
        path: '/reports',
        element: (
            <ProtectedRoute>
                <ReportsPage />
            </ProtectedRoute>
        ),
    },
    {
        path: '*',
        element: <Navigate to="/" replace />,
    },
]);
