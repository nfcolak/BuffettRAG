# BuffettRAG answer-quality hardening

## Scope

Bu çalışma `04f1469e0944b5f56a0ef668aaea94319efc03a5` tabanında, bulut modeli veya gizli anahtar kullanmadan yapıldı. Amaç ölçülmeyen bir "kalite artışı" iddia etmek değil; cevapların yanlış bağlama, eksik dönem kapsamına ve doğrulanamayan citation'a dayanmasına yol açan kök nedenleri azaltmaktır.

Kaynak corpus değişmedi: SHA-256 `cf856872b8014f584c0990f9dcf21eef4b5d43419f672fc5e9aa722b199dd48a`, 48 mektup, 5.833 chunk.

## Yapılan iyileştirmeler

1. **Sıfır kanıtlı BM25 sonuçları engellendi.** BM25'nin küçük koleksiyonlarda sıfır veya negatif IDF üretmesi, RRF ile corpus sırasındaki alakasız kayıtları aday yapabiliyordu. Şimdi adayın sorguyla en az bir gerçek ortak token taşıması gerekiyor.
2. **Genişletilmiş sorgu ile cevap sorgusu ayrıldı.** Query expansion aday bulmak için yardımcıdır; temporal intent ve cross-encoder reranking artık kullanıcının asıl sorusuyla çalışır.
3. **Dönem karşılaştırmaları kapsayıcı hale getirildi.** `2008 vs 2020` gibi açık yıl kıyasları tespit ediliyor; iki dönemden de aday varsa her dönemden en iyi aday için kontenjan ayrılıyor. Tekrarlanan/dengi dönem ifadeleri sahte iki-dönem sorgusuna dönmüyor.
4. **Context sınırları güvenli hale getirildi.** Komşu chunk genişletmesi mektup/yıl sınırını geçmiyor, anchor passage kaybolmuyor, overlap yalnız tam metin çakışmasında temizleniyor ve tüm küçük token bütçelerinde maksimum uzunluğa uyuyor.
5. **Dedup daha ihtiyatlı.** Farklı yıllar, sayılar veya olumsuzluk içeren yakın metinler "duplicate" diye silinmiyor.
6. **Cevap kanıtı görünür ve denetlenebilir.** API artık LLM prompt'una verilen genişletilmiş passages ile ham retrieved passages'i ayırıyor; citation sonuçları `passage_ids` ve `invalid_numbers` içeriyor.
7. **Citation metriği dürüstleştirildi.** Lexical bigram overlap artık entailment/faithfulness diye sunulmuyor. Cümle sonundaki citation'ın sonraki cümleyi kapsaması hatası düzeltildi; answer-evaluation çıktısı prompt evidence, per-sentence sonuçlar ve scoring yöntemi saklıyor.
8. **Refusal davranışı daraltıldı.** Yetersiz retrieved evidence, tüm koleksiyonun sessiz olduğuna dair yanlış bir iddia üretmiyor. Backend refusal aldığında LLM'i tekrar deneyip uydurma cevaba zorlamıyor.

## Ölçüm

Offline kontrol gerçek corpus ve 50 sorguluk mevcut gold set üzerinde çalışır; embedding/reranker indirmez ve bulut LLM çağırmaz. Bu yüzden yalnız BM25 + yerel extractive akış için tanısal bir ölçümdür; production hybrid+reranker answer-quality benchmark değildir.

| Ölçü | Önce | Sonra | Yorum |
|---|---:|---:|---|
| Corpus / soru | 5.833 / 50 | 5.833 / 50 | Aynı sabit veri |
| BM25 MRR (keyword proxy) | 0.6668 | 0.6668 | Bu düzeltmeler BM25 sıralama skorunu oyunlaştırmadı |
| BM25 Recall@1 / @3 / @5 / @8 | 0.58 / 0.70 / 0.80 / 0.84 | 0.58 / 0.70 / 0.80 / 0.84 | Aynı offline benchmark |
| OOV `xyzzyqzzzzz` sonuçları | 10 | 0 | Alakasız sonuç sızıntısı kapandı |
| Context budget aşımı | 0 | 0 | Sınır korunuyor |
| Test suite | mevcut suite | 60 passed | Yeni quality regressions dahil |

Dosyalar:
- Baseline: `data/evaluation/answer_quality_20260910/offline_before.json`
- Sonuç: `data/evaluation/answer_quality_20260910/offline_after.json`
- Corpus audit: `data/evaluation/answer_quality_20260910/corpus_audit.json`

## Bilinen sınır

Tarihi answer artefact'ı 50 girişim içeriyor, ancak yalnız 39'u skorlanmış; 11'i provider rate-limit ile bitmiş. Historical `citation_coverage=0.9124` ve lexical support proxy `0.2543` gerçek answer correctness veya entailment değildir; eski kayıt pasaj metni/chunk ID saklamadığı için sonradan doğrulanamaz.

Bu nedenle bu değişikliklerden sonra "answer quality X puan arttı" iddiası yoktur. Canlı ve insan değerlendirmeli bir benchmark çalıştırılmadan yalnızca yukarıdaki mekanik iyileştirmeler doğrulanmıştır.

## Sonraki en yüksek etkili işler

1. 50 gold soruyu gerçek passage ID, desteklenen claim ve kabul/red ölçütleriyle insan anotasyonuna dönüştürmek.
2. Sabit model/versiyon, sıcaklık, token limiti ve maliyet tavanıyla tekrarlanabilir end-to-end answer benchmark çalıştırmak; provider hatalarını ayrı raporlamak.
3. Citation sonrası NLI entailment + claim splitting eklemek; NLI sonucu yalnız kalite kapısı olarak kullanıp doğruluk etiketi saymamak.
4. PDF yıllarında layout-aware extraction ve başlık/paragraph segmentasyonu yapıp indexi yeniden kurmak. Audit, 1998-2003 ve 2008-2023 PDF'lerinde çok sayıda sentence-end'siz chunk gösteriyor.
5. Retrieval için gerçek hard-negative set oluşturmak; özellikle yakın şirket, dönem ve finans terimlerini ayırt eden sorgular eklemek.
6. Query expansion için structured output, entity/year preservation ve expansion ablation testi eklemek.
7. Reranker ve embedding modelini latency, recall ve citation support üzerinden benchmark etmek; yalnız headline MRR ile seçmemek.
8. Answer generation öncesi evidence sufficiency classifier ve claim-level citation validator eklemek.
9. Production loglarında secret veya ham kullanıcı sorusu tutmadan, request ID, latency, retrieved IDs, refusal nedeni ve evaluator örneklemesi toplamak.
10. Streamlit demo açılırsa, model anahtarını Streamlit Secrets'ta tutmak; ayrı düşük bütçeli anahtar, token/rate limit ve abuse kontrolü uygulamak.

## Tam sıralı kalite programı (2026-09-11)

Bu ek çalışma önceki hardening'i korur; eski `chunks_v2.jsonl` ve önceki raporlar değiştirilmedi. Yeni artefact'lar `data/evaluation/answer_quality_program/` altındadır.

1. **Gerçek cevap benchmark'ı:** `answer_benchmark.json`, 8 insan-kürasyonlu soru için gold passage ID, gerekli claim, kabul eşiği ve açık red ifadelerini saklar. `scripts/eval/run_answer_benchmark.py` answer + citation'ı bu kurallarla deterministik olarak değerlendirir.
2. **Citation sonrası claim kapısı:** `claim_validator.py` citation'lı cümleleri satır/cümle sınırlarında ayırır ve marker'ı claim ile birlikte tutar. NLI yokken lexical overlap tek başına onay vermez: yüksek token kapsamı, polarity, sayı ve entity uyumu birlikte gerekir. Enjekte edilmiş NLI scorer ayrı bir onay yoludur; desteklenmeyen claim cevapta bloklanır.
3. **PDF + paragraph corpus:** `scripts/index/rebuild_paragraph_index.py` 48 ham kaynağı tekrar extract edip `chunks_v3_paragraph.jsonl` üretti. Her kayıt source SHA-256, extractor, paragraph sayısı ve `paragraph_v3` chunker kaynağını taşır. Eski ID'ler sessizce kaybolmadı: `chunk_id_map_v2_to_v3.jsonl` her v2 ID için en çok üç v3 aday ve token-Jaccard skorunu saklar.
4. **Hard negatives:** `hard_negatives_v3.json` 8 yakın/yanıltıcı decoy içerir; yalnız relevant passage named decoy'dan önce gelirse geçer.
5. **Ablation:** `scripts/eval/run_ablation.py`, corpus/model/document-ID ve FAISS artifact hash'lerini doğruladıktan sonra BM25, BGE embedding ve BGE reranker yollarını aynı V3 case setinde çalıştırır.
6. **Structured expansion:** JSON tabanlı terms/entities/years genişletmesi eklendi. Orijinal soru ilk bileşen olarak kalır; yıl ve adlandırılmış entity'ler korunur. Parse hatasında eski comma-list yolu ya da orijinal sorgu kullanılır.
7. **Generation öncesi evidence gate:** pipeline, `/ask` ve `/ask/stream` prompt/LLM çağrısından önce lexical evidence kontrolü yapar. Kanıt yoksa sabit refusal döner; streaming path provider'ı hiç açmaz.
8. **Güvenli Streamlit demo:** `streamlit_app.py`, yalnız `st.secrets` ile HTTPS backend URL ve server-side backend anahtarı okur; kullanıcıdan anahtar istemez. Public backend modu auth, request boyutu, model override, CORS, debug, readiness ve proxy-header güvenini fail-closed doğrular. Şema ve altyapı sınırları: `docs/STREAMLIT_DEPLOYMENT_TR.md`.

## Ölçülen sonuçlar

| Kontrol | V3 sonuç | Yorum |
|---|---:|---|
| Kaynak / paragraph record | 48 / 5.831 | Source SHA ve komşuluk metadata taşır |
| Duplicate ID / hatalı komşu / eksik provenance | 0 / 0 / 0 | `corpus_audit_v3.json` |
| BM25 + embedded extractive kabul | 8/8 | Claim kendi gold citation'ı, polarity ve sayı kontrolleriyle değerlendirilir; canlı LLM değildir |
| Hard-negative BM25 | 8/8 | Relevant passage named decoy'dan önce |
| BM25 Recall@8 | 8/8 | Ortalama 10.5 ms; sekiz-case retrieval check |
| BGE embedding Recall@8 | 6/8 | Ortalama 142.1 ms; local CPU |
| BGE reranker Recall@8 | 7/8 | Ortalama 4.824 s; local CPU |
| Regression suite | 89 passed | Adversarial claim/citation, index hash ve public-demo guard testleri dahil |

Kalan retrieval vakası: reranked BGE yolu `aq05_index` gold passage'ını Recall@8 içinde bulamıyor. Sekiz case küçük ve kürasyonlu bir settir; genel answer-quality iddiası için held-out set ve insan adjudication gerekir.

## Corpus/index durumu

`chunks_v3_paragraph_manifest.json`, corpus üretildiği anda vector index'i `not_built` olarak kaydetti. Daha sonra ayrı offline adımda `data/indices/faiss_v3/` altında 5.831-record FAISS index'i `BAAI/bge-base-en-v1.5` ile üretildi. `index_manifest.json`, corpus SHA-256 `8263f08958729a77febf0922e7bbc4161bedd5fdb3e9a8b3fc9a64ab681ab5f0`, sıralı document-ID hash'i, model/dimension ve `index.faiss`/`meta.json` artifact hash'lerini bağlar; ablation ve runtime yüklemeden önce bunları doğrular. Bu local index `.gitignore` kapsamındadır ve deployment değildir.

## Öncelikli tekrar üretim komutları

```bash
# 1. Tüm regression suite (secret okumaz)
PYTHON_DOTENV_DISABLED=1 python3 -m pytest tests -q

# 2. Paragraph corpus + deterministic eski->yeni ID map
PYTHON_DOTENV_DISABLED=1 python3 scripts/index/rebuild_paragraph_index.py

# 3. V3 corpus integrity/provenance kontrolleri
PYTHON_DOTENV_DISABLED=1 python3 scripts/index/audit_corpus.py --corpus data/processed/chunks_v3_paragraph.jsonl

# 4. Curated V3 offline answer benchmark (cloud çağrısı yok)
PYTHON_DOTENV_DISABLED=1 python3 scripts/eval/run_answer_benchmark.py --corpus data/processed/chunks_v3_paragraph.jsonl --cases data/evaluation/answer_quality_program/answer_benchmark_v3.json --output data/evaluation/answer_quality_program/answer_benchmark_v3_bm25_local.json

# 5. Hard-negative BM25
PYTHON_DOTENV_DISABLED=1 python3 scripts/eval/eval_hard_negatives.py --corpus data/processed/chunks_v3_paragraph.jsonl --cases data/evaluation/answer_quality_program/hard_negatives_v3.json --output data/evaluation/answer_quality_program/hard_negatives_v3_bm25.json

# 6. Offline V3 embedding/reranker ablation (önceden cache'lenmiş modeller)
PYTHON_DOTENV_DISABLED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 scripts/eval/run_ablation.py --corpus data/processed/chunks_v3_paragraph.jsonl --cases data/evaluation/answer_quality_program/answer_benchmark_v3.json --faiss-dir data/indices/faiss_v3 --device cpu --output data/evaluation/answer_quality_program/ablation_v3.json

# 7. Streamlit syntax/safe-no-secrets smoke
PYTHON_DOTENV_DISABLED=1 python3 -m py_compile streamlit_app.py
```

## Açık sınırlamalar ve sonraki adımlar

- Gold benchmark yalnız 8 case içerir; bu set geliştirme sırasında kullanıldığı için genel kalite iddiası için ayrı frozen held-out set, human adjudication ve daha geniş claim annotasyonu gerekir.
- Deterministic claim guard bilinçli olarak ihtiyatlıdır; valid paraphrase'leri bloklayabilir. NLI yolu threshold kalibrasyonu ve human disagreement analizi olmadan production onayı sayılmamalıdır.
- Evidence gate lexical bir sufficiency filtresidir; semantik paraphrase için yeterli değildir.
- BGE reranked yol `aq05_index` vakasını Recall@8 içinde kaçırır; BM25 8/8'dir.
- Live provider answer benchmark ve public deployment yapılmadı. DNS rebinding riski için private/link-local egress firewall, ayrıca TLS, merkezi rate limit, provider spend cap ve monitoring gerekir.
