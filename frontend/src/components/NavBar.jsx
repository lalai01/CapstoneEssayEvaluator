import React, { useEffect, useRef, useState } from 'react';

// Simple outline icons (24×24, stroke-based).
const ICON_PATHS = {
  home: 'M3 11l9-8 9 8M5 10v10h5v-6h4v6h5V10',
  evaluate: 'M4 20h4L19 9l-4-4L4 16zM14 6l4 4',
  compare: 'M3 21h18M6 21V11M11 21V5M16 21v-7M21 21V9',
  knowledge: 'M4 5a2 2 0 012-2h13v16H6a2 2 0 00-2 2zM4 19V5M8 7h7',
  surveys: 'M9 4h6v3H9zM8 5H6v16h12V5h-2M9 12h6M9 16h4',
  rateus: 'M12 3l2.7 5.6 6.1.9-4.4 4.3 1 6.1L12 17l-5.4 2.9 1-6.1L3.2 9.5l6.1-.9z',
  learning: 'M9 18h6M10 21h4M12 3a6 6 0 00-3.5 10.9c.6.4 1 1.1 1 1.8V16h5v-.3c0-.7.4-1.4 1-1.8A6 6 0 0012 3z',
  rubrics: 'M4 17L17 4l3 3L7 20zM8 13l2 2M11 10l2 2M14 7l2 2',
  dashboard: 'M16 20v-1a4 4 0 00-4-4H6a4 4 0 00-4 4v1M9 11a4 4 0 100-8 4 4 0 000 8zM22 20v-1a4 4 0 00-3-3.9M16 3.1a4 4 0 010 7.8',
  school: 'M3 21h18M5 21V10l7-5 7 5v11M9 21v-6h6v6',
  list: 'M8 6h12M8 12h12M8 18h12M4 6h.01M4 12h.01M4 18h.01',
  tag: 'M3 12V4h8l9 9-8 8zM7.5 7.5h.01',
  teacher: 'M12 4l9 4-9 4-9-4zM7 10v5c0 1.7 2.2 3 5 3s5-1.3 5-3v-5M21 8v6',
  shield: 'M12 3l7 3v5c0 4.5-3 8.3-7 10-4-1.7-7-5.5-7-10V6z',
  chevron: 'M6 9l6 6 6-6',
  menu: 'M4 6h16M4 12h16M4 18h16',
  close: 'M6 6l12 12M18 6L6 18',
  logout: 'M15 3h4a2 2 0 012 2v14a2 2 0 01-2 2h-4M10 17l-5-5 5-5M5 12h11',
};

export function Icon({ name, className = 'w-5 h-5' }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75"
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d={ICON_PATHS[name] || ''} />
    </svg>
  );
}

// Closes a menu on outside click or Escape.
function useDismiss(open, setOpen, ref) {
  useEffect(() => {
    if (!open) return undefined;
    const onDown = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    const onKey = (e) => { if (e.key === 'Escape') setOpen(false); };
    document.addEventListener('mousedown', onDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [open, setOpen, ref]);
}

// whitespace-nowrap: a label never breaks onto a second line (which made
// neighbouring labels overlap on mid-size screens).
const linkBase = 'relative inline-flex items-center gap-1.5 xl:gap-2 px-2.5 xl:px-3 py-2 rounded-lg text-sm font-medium whitespace-nowrap transition-colors';
const linkIdle = 'text-slate-300 hover:text-white hover:bg-white/10';
const linkActive = 'text-white bg-white/15 shadow-inner';
// Underline under the current top-bar tab.
const activeBar = 'after:absolute after:left-2.5 after:right-2.5 after:-bottom-[13px] after:h-0.5 after:rounded-full after:bg-indigo-400';

function GroupMenu({ group, items, currentTab, onSelect }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  useDismiss(open, setOpen, ref);
  const active = items.some((i) => i.id === currentTab);

  return (
    <div className="relative" ref={ref}>
      <button
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        className={`${linkBase} ${active ? `${linkActive} ${activeBar}` : linkIdle}`}
      >
        <Icon name={group.icon} className="w-4 h-4" />
        {group.label}
        <Icon name="chevron" className={`w-4 h-4 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>
      {open && (
        <div
          role="menu"
          aria-label={group.label}
          className="absolute right-0 mt-2 w-72 rounded-xl bg-white shadow-2xl ring-1 ring-black/5 p-1.5 z-50"
        >
          <div className="px-3 pt-2 pb-1 text-[11px] font-semibold uppercase tracking-wider text-slate-400">
            {group.caption}
          </div>
          {items.map((item) => (
            <button
              key={item.id}
              type="button"
              role="menuitem"
              onClick={() => { onSelect(item.id); setOpen(false); }}
              className={`w-full flex items-start gap-3 rounded-lg px-3 py-2.5 text-left transition-colors ${
                currentTab === item.id ? 'bg-indigo-50' : 'hover:bg-slate-50'
              }`}
            >
              <span className={`mt-0.5 rounded-md p-1.5 ${currentTab === item.id ? 'bg-indigo-600 text-white' : 'bg-slate-100 text-slate-600'}`}>
                <Icon name={item.icon} className="w-4 h-4" />
              </span>
              <span>
                <span className="block text-sm font-semibold text-slate-800">{item.label}</span>
                {item.description && <span className="block text-xs text-slate-500">{item.description}</span>}
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function AccountMenu({ user, roleBadges, onSignOut }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  useDismiss(open, setOpen, ref);
  const email = user?.email || '';
  const initial = (user?.user_metadata?.full_name || email || '?').trim().charAt(0).toUpperCase();
  const avatar = user?.user_metadata?.avatar_url;

  return (
    <div className="relative" ref={ref}>
      <button
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label="Account menu"
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-2 rounded-full pl-1 pr-2 py-1 hover:bg-white/10 transition-colors"
      >
        {avatar ? (
          <img src={avatar} alt="" referrerPolicy="no-referrer" className="w-8 h-8 rounded-full object-cover ring-2 ring-white/30" />
        ) : (
          <span className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 text-white text-sm font-semibold flex items-center justify-center ring-2 ring-white/30">
            {initial}
          </span>
        )}
        <span className="hidden xl:flex flex-col items-start leading-tight">
          <span className="text-sm text-white max-w-[180px] truncate">{email}</span>
          <span className="text-[11px] text-slate-400">{roleBadges.map((b) => b.label).join(' · ') || 'Loading…'}</span>
        </span>
        <Icon name="chevron" className="w-4 h-4 text-slate-300" />
      </button>
      {open && (
        <div role="menu" aria-label="Account" className="absolute right-0 mt-2 w-72 rounded-xl bg-white shadow-2xl ring-1 ring-black/5 p-1.5 z-50">
          <div className="px-3 py-3 border-b border-slate-100">
            <div className="text-xs text-slate-500">Signed in as</div>
            <div className="text-sm font-semibold text-slate-800 break-all">{email}</div>
            <div className="flex flex-wrap gap-1.5 mt-2">
              {roleBadges.map((b) => (
                <span key={b.label} className={`text-[11px] font-semibold px-2 py-0.5 rounded-full ${b.className}`}>{b.label}</span>
              ))}
            </div>
          </div>
          <button
            type="button"
            role="menuitem"
            onClick={() => { setOpen(false); onSignOut(); }}
            className="w-full flex items-center gap-2 rounded-lg px-3 py-2.5 mt-1 text-sm text-red-600 hover:bg-red-50 transition-colors"
          >
            <Icon name="logout" className="w-4 h-4" />
            Sign out
          </button>
        </div>
      )}
    </div>
  );
}

/**
 * items:   allowed tabs [{ id, label, icon, group, description }]
 * groups:  [{ id, label, caption, icon }] for dropdown groups (not 'main')
 */
export default function NavBar({ items, groups, currentTab, onSelect, user, roleBadges, onSignIn, onSignOut }) {
  const [mobileOpen, setMobileOpen] = useState(false);
  const mainItems = items.filter((i) => i.group === 'main');
  const visibleGroups = groups
    .map((g) => ({ ...g, items: items.filter((i) => i.group === g.id) }))
    .filter((g) => g.items.length > 0);

  const select = (id) => {
    onSelect(id);
    setMobileOpen(false);
  };

  return (
    <header className="sticky top-0 z-40 bg-slate-950/95 md:bg-slate-950/85 md:backdrop-blur-md border-b border-white/10 shadow-lg shadow-black/20">
      <div className="max-w-7xl mx-auto px-4 h-16 flex items-center gap-3">
        <button type="button" onClick={() => select('home')} className="flex items-center gap-3 shrink-0 rounded-lg pr-2" aria-label="AI Essay Evaluator home">
          <span className="w-9 h-9 rounded-xl bg-gradient-to-br from-blue-500 via-indigo-500 to-purple-600 flex items-center justify-center text-white font-bold text-sm shadow-md">
            AE
          </span>
          {/* The name needs room the tabs use on mid-size screens: show it on
              phones (no tabs there) and wide screens only. */}
          <span className="hidden sm:flex lg:hidden xl:flex flex-col items-start leading-tight">
            <span className="text-white font-semibold whitespace-nowrap">AI Essay Evaluator</span>
            <span className="text-[11px] text-slate-400 whitespace-nowrap">Writing assessment platform</span>
          </span>
        </button>

        <nav aria-label="Main" className="hidden lg:flex items-center gap-0.5 xl:gap-1 ml-2 xl:ml-4 min-w-0">
          {mainItems.map((item) => (
            <button
              key={item.id}
              type="button"
              aria-current={currentTab === item.id ? 'page' : undefined}
              onClick={() => select(item.id)}
              title={item.short ? item.label : undefined}
              className={`${linkBase} ${currentTab === item.id ? `${linkActive} ${activeBar}` : linkIdle}`}
            >
              <Icon name={item.icon} className="w-4 h-4 shrink-0" />
              {item.short || item.label}
            </button>
          ))}
        </nav>

        <div className="ml-auto flex items-center gap-1 xl:gap-2 shrink-0">
          <div className="hidden lg:flex items-center gap-0.5 xl:gap-1">
            {visibleGroups.map((g) => (
              <GroupMenu key={g.id} group={g} items={g.items} currentTab={currentTab} onSelect={select} />
            ))}
          </div>

          {user ? (
            <AccountMenu user={user} roleBadges={roleBadges} onSignOut={onSignOut} />
          ) : (
            <button
              type="button"
              onClick={onSignIn}
              className="px-4 py-2 rounded-lg text-sm font-semibold bg-white text-slate-900 hover:bg-indigo-50 transition-colors"
            >
              Sign in
            </button>
          )}

          <button
            type="button"
            className="lg:hidden p-2 rounded-lg text-slate-200 hover:bg-white/10"
            aria-label={mobileOpen ? 'Close menu' : 'Open menu'}
            aria-expanded={mobileOpen}
            onClick={() => setMobileOpen((o) => !o)}
          >
            <Icon name={mobileOpen ? 'close' : 'menu'} className="w-6 h-6" />
          </button>
        </div>
      </div>

      {mobileOpen && (
        <nav aria-label="Main" className="lg:hidden border-t border-white/10 bg-slate-950/95 max-h-[calc(100vh-4rem)] overflow-y-auto">
          <div className="max-w-7xl mx-auto px-4 py-3 space-y-4">
            {[{ id: 'main', label: 'Menu', items: mainItems }, ...visibleGroups].map((section) => (
              <div key={section.id}>
                <div className="px-3 pb-1 text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                  {section.label}
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-1">
                  {section.items.map((item) => (
                    <button
                      key={item.id}
                      type="button"
                      aria-current={currentTab === item.id ? 'page' : undefined}
                      onClick={() => select(item.id)}
                      className={`${linkBase} w-full justify-start ${currentTab === item.id ? linkActive : linkIdle}`}
                    >
                      <Icon name={item.icon} className="w-4 h-4" />
                      {item.label}
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </nav>
      )}
    </header>
  );
}
