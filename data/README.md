Raw and processed corpora for AI-Humanizer.

## Layout

- `raw/<dataset_name>/` — original downloads / exports (not trained on automatically)
- `processed/<dataset_name>/` — normalized rows written via repository helpers

Access data only through:

- `repositories.base.DatasetRepository`
- `data_sources` catalog / source adapters

No MongoDB / PostgreSQL / MySQL.

## Datasets to use (current)

| Need | Dataset | Role in code |
|------|---------|--------------|
| Human writing style (short) | **Blog Authorship Corpus** | `blog_authorship` → `human_style` |
| Human writing style (long) | **PG-19** | `pg19` → `human_style_long` |
| AI vs human detection | **RAID**, **MAGE** | `raid`, `mage` → `ai_vs_human` |
| Paraphrase / meaning keep | **PAR-3** (DIPPER-related) | `par3` → `paraphrase` |
| Rough AI↔human pairs | **HC3** (older ChatGPT) | `hc3` → `ai_human_pairs_rough` |

Catalog: [`configs/datasets.yaml`](../configs/datasets.yaml)  
Adapters: [`data_sources/`](../data_sources/)

Place files under e.g. `data/raw/hc3/*.jsonl` — adapters read JSON / JSONL / CSV / TXT.

## Gap: no ready-made AI → humanized map

There is **no single public dataset** that directly maps:

```
AI-generated text  →  humanized text
```

for this project.

### Planned later (not implemented now)

1. Take **human** text (Blog Authorship / PG-19 / …)
2. Rewrite it with an **AI** model
3. Store pairs as:

```
source_text = AI rewrite   # input to the humanizer
target_text = original human text
```

Tracked in catalog as `future_pair_construction` with `enabled: false`.

Until then, HC3 is only a **rough** paired stopgap; RAID/MAGE support discrimination; PAR-3 supports paraphrase/semantics; Blog/PG-19 support human style.

## Do not train yet

Adding files here does **not** start training. Use `python main.py --dry-run` / `pytest` for verification.
