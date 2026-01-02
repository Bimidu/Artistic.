tailwind.config = {
    theme: {
        extend: {
            fontFamily: {
                sans: ['Saans', 'system-ui', 'sans-serif'],
                heading: ['Saans', 'system-ui', 'sans-serif'],
                mono: ['ui-monospace', 'monospace'],
            },
            colors: {
                primary: {
                    50: '#f8f9fa',
                    100: '#f1f3f5',
                    200: '#e9ecef',
                    300: '#dee2e6',
                    400: '#ced4da',
                    500: '#adb5bd',
                    600: '#868e96',
                    700: '#495057',
                    800: '#343a40',
                    900: '#212529',
                },
                accent: {
                    50: '#e3f2fd',
                    100: '#bbdefb',
                    200: '#90caf9',
                    300: '#64b5f6',
                    400: '#42a5f5',
                    500: '#2196f3',
                    600: '#1e88e5',
                    700: '#1976d2',
                    800: '#1565c0',
                    900: '#0d47a1',
                }
            },
            fontSize: {
                '7xl': '4.5rem',
                '8xl': '6rem',
                '9xl': '8rem',
            }
        }
    }
}
