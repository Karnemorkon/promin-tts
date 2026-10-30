import { NavLink } from 'react-router-dom';
import { BookOpen, LogOut, MousePointerClick } from 'lucide-react';
import { useConnection } from '../stores/connection';

export default function Navbar({ onDisconnect }: { onDisconnect: () => void }) {
  const { state } = useConnection();

  return (
    <nav className="w-60 bg-white border-r border-gray-200 flex flex-col shadow-sm">
      <div className="p-5 border-b border-gray-200">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
            <MousePointerClick className="w-5 h-5 text-white" />
          </div>
          <span className="font-bold text-gray-800 text-sm tracking-tight uppercase">
            Promin TTS
          </span>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto py-4 px-2">
        <NavLink
          to="/projects"
          className={({ isActive }) =>
            `flex items-center gap-3 px-4 py-2.5 text-sm font-medium rounded-lg transition-colors ${
              isActive
                ? 'bg-blue-50 text-blue-700'
                : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
            }`
          }
        >
          <BookOpen className="w-5 h-5" />
          Проєкти
        </NavLink>
      </div>

      <div className="p-4 border-t border-gray-200 space-y-2">
        <div className="text-xs text-gray-500">
          <div className="font-medium text-gray-700">
            {state.mode === 'local' ? 'Локальний сервер' : 'Віддалений сервер'}
          </div>
          <div className="truncate">{state.baseUrl || window.location.origin}</div>
        </div>
        <button
          onClick={onDisconnect}
          className="w-full flex items-center gap-3 px-3 py-2 text-sm text-gray-600 hover:bg-gray-100 hover:text-gray-900 rounded-lg transition-colors"
        >
          <LogOut className="w-4 h-4" />
          Відключитись
        </button>
      </div>
    </nav>
  );
}
