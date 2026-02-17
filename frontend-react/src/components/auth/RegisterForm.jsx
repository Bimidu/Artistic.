import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useForm } from 'react-hook-form';
import { useAuth } from '@hooks/useAuth';
import { Input } from '@components/common/Input';
import { Button } from '@components/common/Button';
import { GoogleAuthButton } from './GoogleAuthButton';
import { validateEmail, validatePassword, getPasswordStrength } from '@utils/validation';

export const RegisterForm = () => {
    const navigate = useNavigate();
    const { register: registerUser } = useAuth();
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);
    const [isAdminSignup, setIsAdminSignup] = useState(false);

    const {
        register,
        handleSubmit,
        watch,
        formState: { errors },
    } = useForm();

    const watchPassword = watch('password', '');
    const passwordStrength = getPasswordStrength(watchPassword);

    const onSubmit = async (data) => {
        setError('');
        setLoading(true);

        try {
            await registerUser(data.name, data.email, data.password, isAdminSignup ? 'admin' : 'user');
            navigate('/');
        } catch (err) {
            setError(err.response?.data?.detail || 'Registration failed');
        } finally {
            setLoading(false);
        }
    };

    return (
        <form onSubmit={handleSubmit(onSubmit)} className={`space-y-6 ${isAdminSignup ? 'p-6 border-2 border-lime-600 rounded-2xl bg-lime-50/30' : ''}`}>
            <h2 className={`text-4xl mb-8 ${isAdminSignup ? 'text-lime-900' : 'text-primary-900'}`}>
                Create Account
            </h2>

            {error && (
                <div className="bg-red-50 border-l-4 border-red-500 p-4 rounded">
                    <p className="text-red-700">{error}</p>
                </div>
            )}

            <Input
                label="Full Name"
                type="text"
                {...register('name', {
                    required: 'Name is required',
                    minLength: { value: 2, message: 'Name must be at least 2 characters' },
                })}
                error={errors.name?.message}
            />

            <Input
                label="Email"
                type="email"
                {...register('email', {
                    required: 'Email is required',
                    validate: (value) => validateEmail(value) || 'Invalid email format',
                })}
                error={errors.email?.message}
            />

            <div>
                <Input
                    label="Password"
                    type="password"
                    showPasswordToggle
                    {...register('password', {
                        required: 'Password is required',
                        validate: (value) => {
                            const validation = validatePassword(value);
                            if (!validation.isValid) {
                                return 'Password must be at least 8 characters with uppercase, lowercase, and number';
                            }
                            return true;
                        },
                    })}
                    error={errors.password?.message}
                />
                {watchPassword && (
                    <div className="mt-2">
                        <div className="flex gap-1 mb-1">
                            <div
                                className={`h-1 flex-1 rounded ${passwordStrength.strength === 'weak'
                                    ? 'bg-red-500'
                                    : passwordStrength.strength === 'medium'
                                        ? 'bg-yellow-500'
                                        : 'bg-green-500'
                                    }`}
                            ></div>
                            <div
                                className={`h-1 flex-1 rounded ${passwordStrength.strength === 'medium' || passwordStrength.strength === 'strong'
                                    ? passwordStrength.strength === 'medium'
                                        ? 'bg-yellow-500'
                                        : 'bg-green-500'
                                    : 'bg-primary-200'
                                    }`}
                            ></div>
                            <div
                                className={`h-1 flex-1 rounded ${passwordStrength.strength === 'strong' ? 'bg-green-500' : 'bg-primary-200'
                                    }`}
                            ></div>
                        </div>
                        <p className="text-sm text-primary-600 capitalize">{passwordStrength.strength} password</p>
                    </div>
                )}
            </div>

            <Input
                label="Confirm Password"
                type="password"
                showPasswordToggle
                {...register('confirmPassword', {
                    required: 'Please confirm your password',
                    validate: (value) => value === watchPassword || 'Passwords do not match',
                })}
                error={errors.confirmPassword?.message}
            />

            <label className="flex items-start gap-3">
                <input
                    type="checkbox"
                    className="w-5 h-5 rounded mt-1 text-primary-600"
                    {...register('terms', {
                        required: 'You must agree to the terms',
                    })}
                />
                <span className="text-sm text-primary-700">
                    I agree to the{' '}
                    <a href="#" className="text-primary-900 hover:underline">
                        Terms of Service
                    </a>{' '}
                    and{' '}
                    <a href="#" className="text-primary-900 hover:underline">
                        Privacy Policy
                    </a>
                </span>
            </label>
            {errors.terms && <p className="text-red-500 text-sm -mt-4">{errors.terms.message}</p>}

            <div className="text-center">
                <button
                    type="button"
                    onClick={() => setIsAdminSignup(!isAdminSignup)}
                    className="text-sm text-primary-700 hover:text-primary-900 transition-colors underline mb-4"
                >
                    {isAdminSignup ? '✓ Sign up as Admin' : 'Sign up as Admin'}
                </button>
            </div>

            <Button type="submit" variant="primary" loading={loading} className="w-full">
                Create Account
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
                Already have an account?{' '}
                <Link to="/login" className="text-primary-900 font-medium hover:underline">
                    Sign in
                </Link>
            </p>
        </form>
    );
};
