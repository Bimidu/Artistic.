import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useForm } from 'react-hook-form';
import { useAuth } from '@hooks/useAuth';
import { Input } from '@components/common/Input';
import { Button } from '@components/common/Button';
import { GoogleAuthButton } from './GoogleAuthButton';
import { validateEmail } from '@utils/validation';

export const LoginForm = () => {
    const navigate = useNavigate();
    const { login } = useAuth();
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);

    const {
        register,
        handleSubmit,
        formState: { errors },
    } = useForm();

    const onSubmit = async (data) => {
        setError('');
        setLoading(true);

        try {
            await login(data.email, data.password);
            navigate('/');
        } catch (err) {
            setError(err.response?.data?.detail || 'Invalid credentials');
        } finally {
            setLoading(false);
        }
    };

    return (
        <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
            <h2 className="text-4xl text-primary-900 mb-8">
                Sign In
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
                label="Password"
                type="password"
                showPasswordToggle
                {...register('password', {
                    required: 'Password is required',
                })}
                error={errors.password?.message}
            />

            <div className="flex items-center justify-between">
                <label className="flex items-center gap-2">
                    <input type="checkbox" className="w-5 h-5 rounded text-primary-600" />
                    <span className="text-sm text-primary-700">Remember me</span>
                </label>
                <Link
                    to="/forgot-password"
                    className="text-sm text-primary-700 hover:text-primary-900 transition-colors"
                >
                    Forgot password?
                </Link>
            </div>

            <Button type="submit" variant="primary" loading={loading} className="w-full">
                Sign In
            </Button>

            <div className="relative my-8">
                <div className="absolute inset-0 flex items-center">
                    <div className="w-full border-t border-primary-200"></div>
                </div>
                <div className="relative flex justify-center text-sm">
                    <span className="px-4 bg-white text-primary-500">Or continue with</span>
                </div>
            </div>

            <GoogleAuthButton />

            <p className="text-center text-primary-600 mt-6">
                Don't have an account?{' '}
                <Link to="/register" className="text-primary-900 font-medium hover:underline">
                    Sign up
                </Link>
            </p>
        </form>
    );
};
