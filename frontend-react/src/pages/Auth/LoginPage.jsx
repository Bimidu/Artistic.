import React from 'react';
import { LoginForm } from '@components/auth/LoginForm';
import { Card } from '@components/common/Card';

export const LoginPage = () => {
    return (
        <div className="min-h-screen bg-gradient-to-b from-primary-100 to-white flex items-center justify-center px-4">
            <div className="max-w-md w-full">
                <div className="text-center mb-8">
                    <h1 className="text-5xl text-primary-900 mb-2">Artistic.</h1>
                    <p className="text-primary-600">Welcome back! Sign in to continue</p>
                </div>
                <Card>
                    <LoginForm />
                </Card>
            </div>
        </div>
    );
};
