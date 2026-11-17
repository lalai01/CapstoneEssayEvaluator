import React, { useEffect, useRef, useState } from 'react';
import toast from 'react-hot-toast';
import { supabase } from '../lib/supabase';
import { useAuth } from '../context/AuthContext';

// Two-factor authentication (TOTP) with an authenticator app such as Google
// Authenticator or Microsoft Authenticator, through Supabase Auth MFA.
// A password (or email link) alone is then not enough to get in: the 6-digit
// code from the phone is needed too. Teachers and admins must use it (the
// backend refuses their powers without it); students can turn it on.

const codeOk = (code) => /^\d{6}$/.test(code);

function CodeInput({ value, onChange, onSubmit, autoFocus = true }) {
  return (
    <input
      type="text"
      inputMode="numeric"
      autoComplete="one-time-code"
      maxLength={6}
      autoFocus={autoFocus}
      value={value}
      onChange={(e) => onChange(e.target.value.replace(/\D/g, '').slice(0, 6))}
      onKeyDown={(e) => { if (e.key === 'Enter' && codeOk(value)) onSubmit(); }}
      aria-label="6-digit code from your authenticator app"
      placeholder="123456"
      className="w-40 rounded-lg border border-gray-300 bg-gray-50 p-2.5 text-center text-xl tracking-[0.4em] font-mono"
    />
  );
}

function friendlyError(error) {
  const msg = error?.message || String(error || '');
  if (/invalid|expired|code/i.test(msg)) return 'That code is wrong or expired. Enter the newest code from the app.';
  if (/rate|too many/i.test(msg)) return 'Too many attempts. Wait a minute and try again.';
  return msg || 'Something went wrong. Please try again.';
}

// Set up: scan the QR code, then confirm with the first code.
export function EnrollTotp({ onDone, onCancel }) {
  const [factor, setFactor] = useState(null);   // { id, qr, secret }
  const [code, setCode] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const started = useRef(false);

  useEffect(() => {
    if (started.current) return;
    started.current = true;
    (async () => {
      // A set-up that was started but never confirmed blocks a new one: remove it.
      const { data: list } = await supabase.auth.mfa.listFactors();
      for (const f of list?.all || []) {
        if (f.factor_type === 'totp' && f.status !== 'verified') {
          await supabase.auth.mfa.unenroll({ factorId: f.id });
        }
      }
      const { data, error: err } = await supabase.auth.mfa.enroll({
        factorType: 'totp',
        friendlyName: `Authenticator ${new Date().toISOString().slice(0, 16)}`,
      });
      if (err) { setError(friendlyError(err)); return; }
      setFactor({ id: data.id, qr: data.totp.qr_code, secret: data.totp.secret });
    })();
  }, []);

  const verify = async () => {
    if (!factor || !codeOk(code)) return;
    setBusy(true);
    setError('');
    const { error: err } = await supabase.auth.mfa.challengeAndVerify({ factorId: factor.id, code });
    setBusy(false);
    if (err) { setError(friendlyError(err)); setCode(''); return; }
    toast.success('Two-factor authentication is on.');
    onDone?.();
  };

  return (
    <div className="space-y-4 text-sm text-gray-700">
      <ol className="list-decimal pl-5 space-y-1">
        <li>Install an authenticator app on your phone (Google Authenticator, Microsoft Authenticator or Authy).</li>
        <li>In the app, add an account and scan this QR code.</li>
        <li>Enter the 6-digit code the app shows.</li>
      </ol>
      {!factor && !error && <p className="text-gray-500">Preparing your QR code…</p>}
      {factor && (
        <div className="flex flex-col items-center gap-2">
          <img src={factor.qr} alt="QR code to scan with your authenticator app" className="h-44 w-44 rounded-lg border bg-white p-2" />
          <details className="text-xs text-gray-500">
            <summary className="cursor-pointer">Can’t scan? Enter this key instead</summary>
            <code className="mt-1 block break-all rounded bg-gray-100 px-2 py-1 font-mono text-gray-800">{factor.secret}</code>
          </details>
          <CodeInput value={code} onChange={setCode} onSubmit={verify} />
        </div>
      )}
      {error && <p className="text-red-600" role="alert">{error}</p>}
      <div className="flex justify-end gap-2">
        {onCancel && (
          <button type="button" onClick={onCancel} className="rounded-xl bg-gray-100 px-4 py-2 font-medium text-gray-800 hover:bg-gray-200">
            Cancel
          </button>
        )}
        <button
          type="button"
          onClick={verify}
          disabled={!factor || !codeOk(code) || busy}
          className="rounded-xl bg-indigo-600 px-4 py-2 font-semibold text-white hover:bg-indigo-700 disabled:opacity-50"
        >
          {busy ? 'Checking…' : 'Turn on'}
        </button>
      </div>
    </div>
  );
}

// Sign-in step two: the code from the authenticator app.
export function ChallengeTotp({ onDone }) {
  const [code, setCode] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  const verify = async () => {
    if (!codeOk(code)) return;
    setBusy(true);
    setError('');
    const { data: list, error: listErr } = await supabase.auth.mfa.listFactors();
    const factor = list?.totp?.[0];
    if (listErr || !factor) {
      setBusy(false);
      setError(friendlyError(listErr || 'No authenticator is set up for this account.'));
      return;
    }
    const { error: err } = await supabase.auth.mfa.challengeAndVerify({ factorId: factor.id, code });
    setBusy(false);
    if (err) { setError(friendlyError(err)); setCode(''); return; }
    onDone?.();
  };

  return (
    <div className="space-y-4 text-sm text-gray-700">
      <p>Open your authenticator app and enter the 6-digit code for AI Essay Evaluator.</p>
      <div className="flex justify-center"><CodeInput value={code} onChange={setCode} onSubmit={verify} /></div>
      {error && <p className="text-red-600" role="alert">{error}</p>}
      <div className="flex justify-end">
        <button
          type="button"
          onClick={verify}
          disabled={!codeOk(code) || busy}
          className="rounded-xl bg-indigo-600 px-4 py-2 font-semibold text-white hover:bg-indigo-700 disabled:opacity-50"
        >
          {busy ? 'Checking…' : 'Verify'}
        </button>
      </div>
    </div>
  );
}

function Modal({ title, subtitle, children, labelId }) {
  return (
    <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/60 p-4 animate-fade-in">
      <div role="dialog" aria-modal="true" aria-labelledby={labelId} className="w-full max-w-md max-h-[92vh] overflow-y-auto rounded-2xl bg-white p-6 shadow-2xl">
        <h3 id={labelId} className="text-lg font-bold text-gray-900">{title}</h3>
        {subtitle && <p className="mt-1 mb-4 text-sm text-gray-600">{subtitle}</p>}
        {children}
      </div>
    </div>
  );
}

// Shown over the whole app while this session still needs a code, or while a
// teacher or admin hasn't set up two-factor authentication yet.
export function TwoFactorGate() {
  const { user, mfa, signOut } = useAuth();
  if (!user || !mfa?.known || (!mfa.needsCode && !mfa.mustEnroll)) return null;
  const done = () => mfa.refresh();
  return (
    <Modal
      labelId="mfa-gate-title"
      title={mfa.needsCode ? '🔐 Enter your verification code' : '🔐 Set up two-factor authentication'}
      subtitle={mfa.needsCode
        ? 'Your account is protected with two-factor authentication.'
        : 'Every account must use two-factor authentication, so a stolen password alone can’t open it. Set it up once (about a minute); after that, each sign-in asks for the 6-digit code from your phone.'}
    >
      {mfa.needsCode ? <ChallengeTotp onDone={done} /> : <EnrollTotp onDone={done} />}
      <p className="mt-4 border-t pt-3 text-xs text-gray-500">
        Signed in as {user.email}.{' '}
        <button type="button" onClick={signOut} className="font-semibold text-indigo-700 underline">Sign out</button>
        {mfa.needsCode && ' · Lost your phone? Ask an admin to reset your two-factor authentication.'}
      </p>
    </Modal>
  );
}

// Account menu → Two-factor authentication: status, turn on, turn off.
export function TwoFactorSettings({ open, onClose }) {
  const { mfa } = useAuth();
  const [setting, setSetting] = useState(false);
  const [busy, setBusy] = useState(false);
  if (!open) return null;

  const turnOff = async () => {
    setBusy(true);
    const { data } = await supabase.auth.mfa.listFactors();
    let failed = null;
    for (const f of data?.totp || []) {
      const { error } = await supabase.auth.mfa.unenroll({ factorId: f.id });
      if (error) failed = error;
    }
    setBusy(false);
    if (failed) { toast.error(friendlyError(failed)); return; }
    await supabase.auth.refreshSession();
    await mfa.refresh();
    toast.success('Two-factor authentication is off.');
  };

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/50 p-4 animate-fade-in" onMouseDown={onClose}>
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="mfa-settings-title"
        className="w-full max-w-md max-h-[92vh] overflow-y-auto rounded-2xl bg-white p-6 shadow-2xl"
        onMouseDown={(e) => e.stopPropagation()}
      >
        <h3 id="mfa-settings-title" className="text-lg font-bold text-gray-900">🔐 Two-factor authentication</h3>
        {setting ? (
          <div className="mt-4">
            <EnrollTotp onDone={async () => { setSetting(false); await mfa.refresh(); }} onCancel={() => setSetting(false)} />
          </div>
        ) : (
          <div className="mt-3 space-y-3 text-sm text-gray-700">
            <p>
              Status:{' '}
              {mfa.enrolled
                ? <strong className="text-emerald-700">On</strong>
                : <strong className="text-gray-800">Off</strong>}
            </p>
            <p>
              With two-factor authentication, signing in needs your password (or email link) <em>and</em> a 6-digit
              code from an authenticator app on your phone, so a stolen password alone can’t open your account.
            </p>
            {mfa.required && (
              <p className="rounded-lg border border-indigo-200 bg-indigo-50 px-3 py-2 text-indigo-900">
                Required for your account, so it can’t be turned off here.
                Every sign-in asks for the code, and you are signed out automatically after a day without activity or a week after signing in.
              </p>
            )}
            <div className="flex justify-end gap-2 pt-2">
              <button type="button" onClick={onClose} className="rounded-xl bg-gray-100 px-4 py-2 font-medium text-gray-800 hover:bg-gray-200">Close</button>
              {!mfa.enrolled && (
                <button type="button" onClick={() => setSetting(true)} className="rounded-xl bg-indigo-600 px-4 py-2 font-semibold text-white hover:bg-indigo-700">
                  Turn on
                </button>
              )}
              {mfa.enrolled && !mfa.required && (
                <button type="button" onClick={turnOff} disabled={busy} className="rounded-xl bg-red-600 px-4 py-2 font-semibold text-white hover:bg-red-700 disabled:opacity-50">
                  {busy ? 'Turning off…' : 'Turn off'}
                </button>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
