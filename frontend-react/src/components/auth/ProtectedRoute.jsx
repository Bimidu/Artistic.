import React from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { useAuth } from '@hooks/useAuth';

export const ProtectedRoute = ({ children }) => {
    const { isAuthenticated, loading } = useAuth();
    const location = useLocation();

    if (loading) {
        return (
            <div className="flex items-center justify-center min-h-screen">
                <div className="spinner w-12 h-12 text-primary-900"></div>
            </div>
        );
    }

    if (!isAuthenticated) {
        return <Navigate to="/login" state={{ from: location }} replace />;
    }

    return children;
};
