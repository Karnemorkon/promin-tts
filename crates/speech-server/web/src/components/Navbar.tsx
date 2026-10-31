import { useEffect, useState } from 'react';
import { NavLink } from 'react-router-dom';
import {
  BookOpen,
  LogOut,
  MousePointerClick,
  PanelLeftClose,
  PanelLeftOpen,
} from 'lucide-react';
import { useConnection } from '../stores/connection';

const COLLAPSE_KEY = 'promin_navbar_collapsed';

export default function Navbar({ onDisconnect }: { onDisconnect: () => void }) {
  const { state } = useConnection();
  const [collapsed, setCollapsed] = useState(() => {
    try {
      return localStorage.getItem(COLLAPSE_KEY) === '1';
    } catch {
      return false;
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem(COLLAPSE_KEY, collapsed ? '1' : '0');
    } catch {
      /* localStorage може бути недоступний у приватному режимі */
    }
  }, [collapsed]);

  return (
    <nav
      className={`${
        collapsed ? 'w-16' : 'w-60'
      } bg-white border-r border-gray-200 flex flex-col shadow-sm transition-all duration-200`}
    >
      <div
        className={`${
          collapsed ? 'px-2 py-4' : 'p-5'
        } border-b border-gray-200`}
      >
        <div
          className={`flex items-center ${
            collapsed ? 'justify-center' : 'gap-2'
          }`}
        >
          <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center shrink-0">
            <MousePointerClick className="w-5 h-5 text-white" />
          </div>
          {!collapsed && (
            <span className="font-bold text-gray-800 text-sm tracking-tight uppercase">
              Promin TTS
            </span>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto py-4 px-2">
        <NavLink
          to="/projects"
          title={collapsed ? 'Проєкти' : undefined}
          className={({ isActive }) =>
            `flex items-center ${
              collapsed ? 'justify-center px-2' : 'gap-3 px-4'
            } py-2.5 text-sm font-medium rounded-lg transition-colors ${
              isActive
                ? 'bg-blue-50 text-blue-700'
                : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
            }`
          }
        >
          <BookOpen className="w-5 h-5 shrink-0" />
          {!collapsed && <span>Проєкти</span>}
        </NavLink>
      </div>

      <div
        className={`${
          collapsed ? 'p-2' : 'p-4'
        } border-t border-gray-200 space-y-2`}
      >
        {!collapsed && (
          <div className="text-xs text-gray-500">
            <div className="font-medium text-gray-700">
              {state.mode === 'local' ? 'Локальний сервер' : 'Віддалений сервер'}
            </div>
            <div className="truncate">{state.baseUrl || window.location.origin}</div>
          </div>
        )}
        <button
          onClick={onDisconnect}
          title="Відключитись"
          className={`w-full flex items-center ${
            collapsed ? 'justify-center px-2' : 'gap-3 px-3'
          } py-2 text-sm text-gray-600 hover:bg-gray-100 hover:text-gray-900 rounded-lg transition-colors`}
        >
          <LogOut className="w-4 h-4 shrink-0" />
          {!collapsed && <span>Відключитись</span>}
        </button>
        <button
          onClick={() => setCollapsed((value) => !value)}
          title={collapsed ? 'Розгорнути меню' : 'Згорнути меню'}
          className={`w-full flex items-center ${
            collapsed ? 'justify-center px-2' : 'gap-3 px-3'
          } py-2 text-sm text-gray-600 hover:bg-gray-100 hover:text-gray-900 rounded-lg transition-colors`}
        >
          {collapsed ? (
            <PanelLeftOpen className="w-4 h-4 shrink-0" />
          ) : (
            <PanelLeftClose className="w-4 h-4 shrink-0" />
          )}
          {!collapsed && <span>Згорнути меню</span>}
        </button>
      </div>
    </nav>
  );
}
