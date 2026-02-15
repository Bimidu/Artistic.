import React from 'react';
import { RegisterForm } from '@components/auth/RegisterForm';
import { Card } from '@components/common/Card';

export const RegisterPage = () => {
    return (
        <div className="min-h-screen bg-gradient-to-b from-primary-100 to-white flex items-center justify-center px-4 py-12">
            <div className="max-w-md w-full">
                <div className="text-center mb-8">
                    <h1 className="text-5xl text-primary-900 mb-2">ASD Detection</h1>
                    <p className="text-primary-600">Create your account to get started</p>
                </div>
                <Card>
                    <RegisterForm />
                </Card>
            </div>
        </div>
    );
};
