import React, { useState } from 'react';
import toast from 'react-hot-toast';
import { declareAge } from '../api';
import { useAuth } from '../context/AuthContext';
import { errorMessage } from '../lib/scoring';

// Asked once, after signing in (students only): are you 18 or older? Under
// the Data Privacy Act (RA 10173) a minor can't consent alone, so a minor's
// account waits until a teacher or admin confirms the parent's or guardian's
// consent. Only the answer is kept, not the birth date.
export default function AgeQuestion() {
  const { user, refreshMe, signOut } = useAuth();
  const [busy, setBusy] = useState(null);

  const answer = async (under18) => {
    setBusy(under18 ? 'minor' : 'adult');
    try {
      await declareAge(under18);
      if (under18) toast('A teacher or admin will confirm your parent’s or guardian’s consent before you can evaluate essays.', { icon: '🧒', duration: 8000 });
      refreshMe();
    } catch (err) {
      toast.error(errorMessage(err));
      setBusy(null);
    }
  };

  return (
    <div className="mx-auto w-full max-w-md rounded-2xl bg-white p-6 shadow-2xl animate-fade-in">
      <h1 className="text-xl font-bold text-gray-900">🎂 One quick question</h1>
      <p className="mt-1 text-sm text-gray-600">Are you 18 years old or older?</p>
      <div className="mt-5 grid grid-cols-1 gap-3 sm:grid-cols-2">
        <button
          type="button"
          disabled={!!busy}
          onClick={() => answer(false)}
          className="rounded-xl bg-indigo-600 px-4 py-3 font-semibold text-white hover:bg-indigo-700 disabled:opacity-50"
        >
          {busy === 'adult' ? 'Saving…' : 'Yes, I’m 18 or older'}
        </button>
        <button
          type="button"
          disabled={!!busy}
          onClick={() => answer(true)}
          className="rounded-xl border border-sky-300 bg-sky-50 px-4 py-3 font-semibold text-sky-900 hover:bg-sky-100 disabled:opacity-50"
        >
          {busy === 'minor' ? 'Saving…' : 'No, I’m under 18'}
        </button>
      </div>
      <div className="mt-5 rounded-lg bg-gray-50 p-3 text-xs text-gray-600 space-y-1">
        <p>
          <strong>Why we ask:</strong> under the Data Privacy Act of 2012 (RA 10173), a student under 18 needs a
          parent’s or guardian’s consent before their essays are processed. If you’re under 18, your teacher confirms
          that your school has that consent; until then you can look around but not evaluate essays.
        </p>
        <p>Only your answer is kept, not your birth date. Answer honestly: a teacher can correct it.</p>
      </div>
      <p className="mt-4 border-t pt-3 text-xs text-gray-500">
        Signed in as {user?.email}.{' '}
        <button type="button" onClick={() => signOut()} className="font-semibold text-indigo-700 underline">Sign out</button>
      </p>
    </div>
  );
}
