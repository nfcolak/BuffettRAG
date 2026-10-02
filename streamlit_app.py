"""Secure, deployment-oriented Streamlit demo for BuffettRAG.

The app never asks visitors for any model settings (answers come from the
backend's embedded model) and only reads deployment values from st.secrets. Keep the backend API key server-side in Streamlit Secrets.
"""
from __future__ import annotations

import uuid

import requests
import streamlit as st

from src.services.demo_security import (
    DemoRequestBudget,
    estimate_token_reservation,
    render_evidence_html,
    validate_demo_backend_url,
)


def _secret(name: str, default):
    """Secrets are optional for a safe, non-configured demo screen."""
    try:
        return st.secrets.get(name, default)
    except st.errors.StreamlitSecretNotFoundError:
        return default


st.set_page_config(page_title="BuffettRAG Evidence Desk", page_icon="📚", layout="wide")
st.markdown("""
<style>
:root { --ink:#17212b; --paper:#f7f4ec; --line:#b9b09e; --accent:#9a3e27; }
.stApp { background:var(--paper); color:var(--ink); }
.block-container { max-width: 1080px; padding-top: 3rem; }
.evidence-title { font-family: Georgia, serif; font-size: 3rem; letter-spacing:-.04em; margin-bottom:0; }
.evidence-kicker { font-family: monospace; color:var(--accent); letter-spacing:.08em; font-size:.78rem; }
.evidence-card { border-top: 1px solid var(--line); padding: 1rem 0; }
</style>
""", unsafe_allow_html=True)

def _allowed_hosts():
    raw = _secret("backend_allowed_hosts", [])
    if isinstance(raw, str):
        raw = raw.split(",")
    return [str(host).strip() for host in raw if str(host).strip()]


allowed_backend_hosts = _allowed_hosts()
backend_url = validate_demo_backend_url(
    str(_secret("backend_url", "")), allowed_hosts=allowed_backend_hosts
)
backend_api_key = str(_secret("backend_api_key", ""))
max_query_chars = min(max(int(_secret("max_query_chars", 1200)), 100), 2000)
request_limit = min(max(int(_secret("request_limit", 12)), 1), 30)
token_limit = min(max(int(_secret("token_limit", 12000)), 2500), 12000)
context_token_allowance = min(max(int(_secret("context_token_allowance", 1500)), 500), 3000)

if "demo_session_id" not in st.session_state:
    st.session_state.demo_session_id = uuid.uuid4().hex
if "demo_budget" not in st.session_state:
    st.session_state.demo_budget = DemoRequestBudget(max_requests=request_limit, max_tokens=token_limit)

st.markdown('<div class="evidence-kicker">BERKSHIRE SHAREHOLDER LETTERS · EVIDENCE-FIRST DEMO</div>', unsafe_allow_html=True)
st.markdown('<div class="evidence-title">BuffettRAG Evidence Desk</div>', unsafe_allow_html=True)
st.caption("Cevaplar kaynak işaretleri ve otomatik kanıt ön kontrolleriyle sunulur; bu kontrol tek başına entailment garantisi değildir. Demo, ziyaretçiden anahtar istemez.")

query = st.text_area("Sorunuzu yazın", max_chars=max_query_chars, height=110,
                     placeholder="Örn. Buffett, 2008 finansal krizinde Berkshire'ın rolünü nasıl açıkladı?")
if st.button("Kanıtla ara ve yanıtla", type="primary", disabled=not backend_url):
    estimated_tokens = estimate_token_reservation(
        query, max_output_tokens=500, context_token_allowance=context_token_allowance
    )
    if not query.strip():
        st.warning("Soru boş olamaz.")
    elif not st.session_state.demo_budget.allow(st.session_state.demo_session_id, tokens=estimated_tokens):
        st.warning("Bu demo oturumu için istek veya token bütçesi doldu. Daha sonra yeniden deneyin.")
    else:
        headers = {"Content-Type": "application/json"}
        if backend_api_key:
            headers["X-API-Key"] = backend_api_key
        try:
            response = requests.post(f"{backend_url}/ask", json={"query": query.strip(), "top_k": 6,
                                     "fetch_k": 30, "rerank": True, "expand_query": False, "max_new_tokens": 500},
                                     headers=headers, timeout=(5, 25), allow_redirects=False)
            if response.status_code == 429:
                st.warning("Sunucu şu anda yoğun; lütfen kısa süre sonra yeniden deneyin.")
            elif response.status_code in (401, 403):
                st.error("Demo hizmeti şu anda yetkilendirilemedi. Yönetici yapılandırmasını kontrol etmelidir.")
            elif response.status_code != 200:
                st.error("Kanıt servisi şu anda yanıt veremiyor. Lütfen daha sonra tekrar deneyin.")
            else:
                payload = response.json()
                answer = payload.get("answer") or "Yeterli kanıt bulunamadı."
                st.subheader("Yanıt")
                st.markdown(answer)
                with st.expander("Kullanılan kanıt pasajları", expanded=True):
                    for index, hit in enumerate(payload.get("hits", []), 1):
                        st.markdown(render_evidence_html(hit, index), unsafe_allow_html=True)
        except (requests.RequestException, ValueError):
            st.error("Kanıt servisine güvenli bir bağlantı kurulamadı. Lütfen daha sonra yeniden deneyin.")

if not backend_url:
    st.info("Demo yapılandırılmadı: geçerli HTTPS backend_url Streamlit Secrets içinde tanımlanmalıdır.")

st.divider()
st.caption("Güvenlik: yalnız HTTPS backend · ziyaretçi anahtarı yok · oturum başına istek/token bütçesi · hata ayrıntıları gizlenir.")
