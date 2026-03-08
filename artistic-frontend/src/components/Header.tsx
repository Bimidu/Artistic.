'use client';

import { useRouter } from 'next/navigation';
import { useAuth } from '@/contexts/AuthContext';

interface HeaderProps {
  showModeToggle: boolean;
  apiUrl: string;
  onShowAuthModal: () => void;
  onShowProfileSidebar: () => void;
}

export default function Header({ showModeToggle, apiUrl, onShowAuthModal, onShowProfileSidebar }: HeaderProps) {
  const router = useRouter();
  const { user } = useAuth();

  return (
    <header className="relative bg-cover bg-center bg-no-repeat" style={{ backgroundImage: 'url(/images/navbar_bg.jpg)' }}>
      <div className="absolute inset-0 bg-primary-900/10" aria-hidden="true" />
      <div className="relative z-10 px-8 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3 flex-nowrap justify-start">
            <div className="text-3xl font-medium text-black tracking-tight whitespace-nowrap">Artistic.</div>
            <div className="hidden lg:block h-6 w-px bg-primary-700 flex-shrink-0"></div>
            <div className="hidden lg:block text-sm text-primary-900 whitespace-nowrap">Speech Analysis Platform</div>
          </div>

          <div className="flex items-center gap-6">
            {showModeToggle && (
              <div className="toggle-switch" id="modeToggle">
                <div className="toggle-option active" data-mode="user">User Mode</div>
                <div className="toggle-option" data-mode="training">Training Mode</div>
                <div className="toggle-slider" id="toggleSlider"></div>
              </div>
            )}

            <div className="flex items-center gap-2 text-sm text-primary-700">
              <span className="w-2 h-2 rounded-full bg-red-400" id="statusDot"></span>
              <span id="statusText">Disconnected</span>
            </div>

            <button
              onClick={() => router.push('/how-it-works')}
              className="px-4 py-2 text-sm bg-white/20 text-primary-900 border border-primary-300 rounded-full font-semibold hover:bg-white hover:text-black transition-all"
            >
              How It Works
            </button>

            <button
              onClick={() => router.push('/guideline')}
              className="px-4 py-2 text-sm bg-black text-primary-400 rounded-full font-bold hover:border-primary-500 hover:text-white transition-all"
            >
              Feature Guide
            </button>

            {/* Authentication UI */}
            {user ? (
              <button
                onClick={onShowProfileSidebar}
                className="flex items-center gap-2 px-3 py-2 bg-white/90 backdrop-blur-sm border border-primary-300 rounded-full hover:bg-white hover:shadow-sm transition-all"
              >
                <span className="text-sm font-medium text-primary-900 hidden sm:inline">
                  {user.full_name}
                </span>
                {user.avatar_url && (
                  <img
                    src={user.avatar_url}
                    alt={user.full_name}
                    className="w-8 h-8 rounded-full border-2 border-primary-200"
                  />
                )}
              </button>
            ) : (
              <button
                onClick={onShowAuthModal}
                className="px-5 py-2 text-sm bg-primary-900 text-white rounded-full font-semibold hover:bg-primary-800 transition-all"
              >
                Login
              </button>
            )}
          </div>
        </div>
      </div>

      {/* API Configuration Bar */}
      <div className="relative z-10 bg-white hidden" id="apiConfigBar">
        <div className="px-8 py-2">
          <div className="flex items-center gap-6 justify-between">
            <div className="flex items-center gap-4 flex-1">
              <label className="text-sm text-primary-600 whitespace-nowrap">API URL</label>
              <input type="text" className="flex-1 px-4 py-2 bg-primary-50 rounded-lg text-sm focus:outline-none focus:bg-primary-100 transition-all" id="apiUrl" defaultValue={apiUrl} />
            </div>
            <button className="px-5 py-2 bg-primary-900 text-white rounded-lg text-sm hover:bg-primary-800 transition-all" onClick={() => (window as Window & { testConnection?: () => void }).testConnection?.()}>Test Connection</button>
          </div>
        </div>
      </div>
    </header>
  );
}
