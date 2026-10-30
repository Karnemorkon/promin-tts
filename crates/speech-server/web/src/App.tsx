import { useEffect, useState } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import Connection from './pages/Connection';
import Projects from './pages/Projects';
import ProjectEditor from './pages/ProjectEditor';
import Navbar from './components/Navbar';
import { ConnectionProvider, useConnection } from './stores/connection.tsx';
import { healthz } from './services/api';

export default function App() {
  return (
    <ConnectionProvider>
      <AppInner />
    </ConnectionProvider>
  );
}

function AppInner() {
  const { state, setState } = useConnection();
  // Якщо підключення відновлено з localStorage — перевіряємо його у фоні.
  const [checking, setChecking] = useState(state.connected);

  useEffect(() => {
    if (!state.connected) return;
    let cancelled = false;
    healthz()
      .then(() => {
        if (!cancelled) setChecking(false);
      })
      .catch(() => {
        if (!cancelled) {
          setChecking(false);
          setState({ connected: false });
        }
      });
    return () => {
      cancelled = true;
    };
    // Перевіряємо лише під час старту.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (checking) {
    return (
      <div className="flex h-screen items-center justify-center bg-gray-100">
        <div className="text-center">
          <div className="w-12 h-12 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-gray-600 text-sm">Підключення до сервера...</p>
        </div>
      </div>
    );
  }

  // Якщо підключення не налаштоване — показуємо екран вибору режиму.
  if (!state.connected) {
    return <Connection onConnect={setState} />;
  }

  return (
    <div className="flex h-screen">
      <Navbar onDisconnect={() => setState({ connected: false })} />
      <div className="flex-1 overflow-hidden flex flex-col">
        <Routes>
          <Route path="/" element={<Navigate to="/projects" replace />} />
          <Route path="/projects" element={<Projects />} />
          <Route path="/projects/:id" element={<ProjectEditor />} />
        </Routes>
      </div>
    </div>
  );
}
