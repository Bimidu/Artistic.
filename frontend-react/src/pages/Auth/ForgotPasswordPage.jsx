import React, { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useForm } from 'react-hook-form';
import { Card } from '@components/common/Card';
import { Input } from '@components/common/Input';
import { Button } from '@components/common/Button';
import { validateEmail } from '@utils/validation';
import api from '@services/api';

export const ForgotPasswordPage = () => {
    const navigate = useNavigate();
    const location = useLocation();
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);
    const [success, setSuccess] = useState(false);

    // Get email from location state if passed from login page
    const emailFromLogin = location.state?.email || '';

    const {
        register,
        handleSubmit,
        watch,
        formState: { errors },
    } = useForm({
        defaultValues: {
            email: emailFromLogin,
        },
    });

    const password = watch('password');

    const onSubmit = async (data) => {
        setError('');
        setLoading(true);

        try {
            await api.post('/auth/reset-password', {
                email: data.email,
                new_password: data.password,
            });

            setSuccess(true);
            setTimeout(() => {
                navigate('/login');
            }, 2000);
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to reset password');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-b from-primary-100 to-white flex items-center justify-center px-4">
            <div className="max-w-md w-full">
                <div className="text-center mb-8">
                    <h1 className="text-5xl text-primary-900 mb-2">Artistic.</h1>
                    <p className="text-primary-600">Reset your password</p>
                </div>
                <Card>
                    {success ? (
                        <div className="text-center py-8">
                            <div className="bg-green-50 border-l-4 border-green-500 p-4 rounded mb-4">
                                <p className="text-green-700">
                                    Password reset successful! Redirecting to login...
                                </p>
                            </div>
                        </div>
                    ) : (
                        <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
                            <h2 className="text-4xl text-primary-900 mb-8">
                                Reset Password
                            </h2>

                            {error && (
                                <div className="bg-red-50 border-l-4 border-red-500 p-4 rounded">
                                    <p className="text-red-700">{error}</p>
                                </div>
                            )}

                            <Input
                                label="Email"
                                type="email"
                                {...register('email', {
                                    required: 'Email is required',
                                    validate: (value) => validateEmail(value) || 'Invalid email format',
                                })}
                                error={errors.email?.message}
                            />

                            <Input
                                label="New Password"
                                type="password"
                                showPasswordToggle
                                {...register('password', {
                                    required: 'Password is required',
                                    minLength: {
                                        value: 8,
                                        message: 'Password must be at least 8 characters',
                                    },
                                })}
                                error={errors.password?.message}
                            />

                            <Input
                                label="Confirm Password"
                                type="password"
                                showPasswordToggle
                                {...register('confirmPassword', {
                                    required: 'Please confirm your password',
                                    validate: (value) =>
                                        value === password || 'Passwords do not match',
                                })}
                                error={errors.confirmPassword?.message}
                            />

                            <Button type="submit" variant="primary" loading={loading} className="w-full">
                                Reset Password
                            </Button>

                            <p className="text-center text-primary-600 mt-6">
                                Remember your password?{' '}
                                <button
                                    type="button"
                                    onClick={() => navigate('/login')}
                                    className="text-primary-900 font-medium hover:underline"
                                >
                                    Sign in
                                </button>
                            </p>
                        </form>
                    )}
                </Card>
            </div>
        </div>
    );
};
