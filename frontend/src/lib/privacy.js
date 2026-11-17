// Data Privacy Act consent, remembered per browser (see PrivacyNotice.jsx).

const CONSENT_KEY = 'essay-evaluator:privacy-consent:v1';

export function hasPrivacyConsent() {
  try {
    return localStorage.getItem(CONSENT_KEY) === 'yes';
  } catch {
    return false;
  }
}

export function saveConsent(agreed) {
  try {
    if (agreed) localStorage.setItem(CONSENT_KEY, 'yes');
    else localStorage.removeItem(CONSENT_KEY);
  } catch {
    // Storage blocked: the box just has to be ticked again next time.
  }
}
