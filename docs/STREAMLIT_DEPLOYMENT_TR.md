# Streamlit deployment (güvenli demo)

## Amaç

`streamlit_app.py`, BuffettRAG backend'e yalnızca HTTPS üzerinden bağlanan, ziyaretçiden API/LLM anahtarı istemeyen demo katmanıdır. Anahtar varsa yalnız Streamlit Secrets'tan okunur ve browser'a ya da loga yazılmaz.

## Secrets şeması

Deployment panelindeki **Secrets** alanına aşağıdaki şemayı kullanın. Değerleri burada, repoda veya sohbet içinde saklamayın:

```toml
backend_url = "https://rag-backend.example.org"
backend_allowed_hosts = ["rag-backend.example.org"]
backend_api_key = "<ayri-dusuk-kotalı-backend-anahtari>"
max_query_chars = 1200
request_limit = 12
token_limit = 12000
context_token_allowance = 1500
```

- `backend_url`: yalnız HTTPS ve 443 portu kabul edilir; hostname `backend_allowed_hosts` içinde olmalı ve çözümleme anında yalnız global IP'lere gitmelidir.
- `backend_allowed_hosts`: dar allowlist'tir; `backend_url` hostunu tam olarak içermelidir. Localhost, private/link-local IP ve `.localhost` adları reddedilir.
- `backend_api_key`: public demoda zorunludur. Bu ayrı düşük-kotalı anahtar yalnız sunucu tarafında `X-API-Key` header'ına gider.
- Her istek için sorgu tahmini + `context_token_allowance` + 500 output token önceden rezerve edilir. Limitler üst sınırlara sıkıştırılır (`max_query_chars≤2000`, `request_limit≤30`, `token_limit≤12000`, `context_token_allowance≤3000`).

**Kesinlikle oluşturmayın/commit etmeyin:** `.streamlit/secrets.toml`. `.gitignore` bunu dışlar.

## Çalıştırma

```bash
PYTHON_DOTENV_DISABLED=1 python -m streamlit run streamlit_app.py
```

Uygulama geçerli secret olmadan güvenli şekilde "yapılandırılmadı" mesajı verir. Bu çalışma canlı deployment veya üretim anahtarı istemez; deployment öncesi ayrı düşük-kotalı backend anahtarı ve HTTPS endpoint sahibi tarafından sağlanmalıdır.

## Backend public-demo ayarları

```bash
PUBLIC_DEMO_MODE=1
API_KEYS=<ayri-dusuk-kotalı-backend-anahtari>
ALLOW_LLM_REQUEST_OVERRIDES=0
TRUST_PROXY_HEADERS=0
EXPOSE_DEBUG_STATUS=0
CORS_ORIGINS=
MAX_REQUEST_BODY_BYTES=1048576
```

Public mod, anahtar yoksa veya debug/trusted-proxy/güvensiz CORS ayarı varsa startup sırasında kapanır. İstemci provider, model veya LLM anahtarı seçemez. `/health` yalnız liveness; `/ready` index/corpus/retriever tutarlılığını bildirir.

## Operasyonel sınırlar

- Streamlit oturum bütçesi yalnız ikincil kullanıcı deneyimi sınırıdır; yeni oturumla aşılabildiği için ana abuse kontrolü değildir.
- Backend API key ve process-local rate limit uygular. Çoklu worker/replica için edge veya ortak Redis rate limit zorunludur.
- Host allowlist ve ilk DNS çözümlemesindeki global-IP kontrolü tek başına DNS rebinding/TOCTOU garantisi vermez. Public deployment'ta backend egress'i private, link-local, metadata ve control-plane ağlarına firewall ile kapatılmalıdır.
- TLS termination, body/connection timeout, WAF, provider spend cap/alert, merkezi redacted log ve secret rotation deployment altyapısında ayrıca doğrulanmalıdır.
- Hata mesajları provider/anahtar ayrıntısı döndürmez; kullanıcı soruları ve secret değerleri loglanmaz.
- SSE yolu ham LLM tokenlarını göndermez; yalnız claim validation sonrasındaki `done` cevabı yayınlanır. Bu, progresif token akışı yerine doğrulanabilir son cevabı tercih eder.
