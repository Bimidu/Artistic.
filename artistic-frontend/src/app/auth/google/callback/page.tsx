'use client';

import { useEffect, Suspense } from 'react';
import { useSearchParams } from 'next/navigation';

function GoogleCallbackContent() {
  const searchParams = useSearchParams();

  useEffect(() => {
    const token = searchParams.get('token');
    console.log('[OAuth Callback] Token received:', token ? 'YES' : 'NO');

    if (!token) {
      console.error('[OAuth Callback] No token in URL params');
      window.location.href = '/?error=no_token';
      return;
    }

    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

    localStorage.setItem('authToken', token);

    console.log('[OAuth Callback] Fetching user profile...');
    fetch(`${apiUrl}/auth/me`, {
      headers: { 'Authorization': `Bearer ${token}` },
    })
      .then((res) => {
        console.log('[OAuth Callback] /auth/me status:', res.status);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((user) => {
        console.log('[OAuth Callback] User profile fetched, storing and redirecting...');
        localStorage.setItem('authUser', JSON.stringify(user));
        window.location.href = '/';
      })
      .catch((error) => {
        console.error('[OAuth Callback] Failed to fetch user profile:', error);
        window.location.href = '/';
      });
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
