export const validateEmail = (email) => {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(email);
};

export const validatePassword = (password) => {
    // Min 8 chars, at least 1 uppercase, 1 lowercase, 1 number
    const minLength = password.length >= 8;
    const hasUppercase = /[A-Z]/.test(password);
    const hasLowercase = /[a-z]/.test(password);
    const hasNumber = /[0-9]/.test(password);

    return {
        isValid: minLength && hasUppercase && hasLowercase && hasNumber,
        minLength,
        hasUppercase,
        hasLowercase,
        hasNumber,
    };
};

export const getPasswordStrength = (password) => {
    const validation = validatePassword(password);
    const score = [
        validation.minLength,
        validation.hasUppercase,
        validation.hasLowercase,
        validation.hasNumber,
    ].filter(Boolean).length;

    if (score <= 1) return { strength: 'weak', color: 'red' };
    if (score <= 3) return { strength: 'medium', color: 'yellow' };
    return { strength: 'strong', color: 'green' };
};

export const validateName = (name) => {
    return name && name.trim().length >= 2;
};
