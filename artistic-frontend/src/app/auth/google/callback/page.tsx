'use client';

import { useEffect, Suspense } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';

function GoogleCallbackContent() {
  const router = useRouter();
  const searchParams = useSearchParams();

  useEffect(() => {
    const token = searchParams.get('token');
    console.log('[OAuth Callback] Token received:', token ? 'YES' : 'NO');

    if (token) {
      console.log('[OAuth Callback] Storing token and redirecting...');

      // Store token immediately
      localStorage.setItem('authToken', token);

      // Fetch user details from backend
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      console.log('[OAuth Callback] Fetching user from:', `${apiUrl}/auth/me`);

      fetch(`${apiUrl}/auth/me`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      })
        .then((res) => {
          console.log('[OAuth Callback] Response status:', res.status);
          if (!res.ok) {
            throw new Error(`HTTP ${res.status}: ${res.statusText}`);
          }
          return res.json();
        })
        .then((user) => {
          console.log('[OAuth Callback] User data received:', user);
          // Store user data
          localStorage.setItem('authUser', JSON.stringify(user));

          console.log('[OAuth Callback] Redirecting to home...');
          // Redirect to home page
          window.location.href = '/';
        })
        .catch((error) => {
          console.error('[OAuth Callback] Error:', error);
          window.location.href = '/?error=auth_failed';
        });
    } else {
      console.log('[OAuth Callback] No token found, redirecting with error');
      window.location.href = '/?error=no_token';
    }
  }, [searchParams]);

  return (
    <div className="min-h-screen flex items-center justify-center bg-white">
      <div className="text-center">
        <div className="w-16 h-16 border-4 border-primary-200 border-t-primary-600 rounded-full animate-spin mx-auto mb-4"></div>
        <h2 className="text-xl font-medium text-primary-900 mb-2">Completing sign in...</h2>
        <p className="text-sm text-primary-500">Please wait while we authenticate your account</p>
      </div>
    </div>
  );
}

export default function GoogleCallback() {
  return (
    <Suspense fallback={
      <div className="min-h-screen flex items-center justify-center bg-white">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-primary-200 border-t-primary-600 rounded-full animate-spin mx-auto mb-4"></div>
          <h2 className="text-xl font-medium text-primary-900 mb-2">Loading...</h2>
        </div>
      </div>
    }>
      <GoogleCallbackContent />
    </Suspense>
  );
}
