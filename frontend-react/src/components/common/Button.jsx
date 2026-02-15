import React from 'react';

export const Button = ({
    children,
    variant = 'primary',
    size = 'default',
    loading = false,
    disabled = false,
    className = '',
    ...props
}) => {
    const baseClasses = 'rounded-2xl font-medium transition-all disabled:opacity-40 disabled:cursor-not-allowed';

    const sizes = {
        sm: 'px-5 py-2 text-sm',
        default: 'px-8 py-4 text-lg',
        lg: 'px-8 py-5 text-xl',
    };

    const variants = {
        primary: 'bg-primary-900 text-white hover:bg-primary-800',
        secondary: 'bg-primary-100 text-primary-900 hover:bg-primary-200',
        outline: 'border-2 border-primary-300 text-primary-900 hover:bg-primary-50',
        ghost: 'text-primary-700 hover:text-primary-900 hover:bg-primary-50',
        headerLight: 'bg-white text-primary-900 hover:bg-primary-100',
    };

    return (
        <button
            className={`${baseClasses} ${sizes[size]} ${variants[variant]} ${className}`}
            disabled={disabled || loading}
            {...props}
        >
            {loading ? (
                <span className="flex items-center justify-center gap-2">
                    <span className="spinner w-5 h-5"></span>
                    Loading...
                </span>
            ) : (
                children
            )}
        </button>
    );
};
