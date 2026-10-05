# 数据说明（data/）

本目录收录论文《文本表征的价值观——国家、媒体与大众三层话语中的中西文化差异测量研究》的**分析级配对语料**（可直接用于复现论文全部测量）。数据为研究用途构建，来源均为公开语料，已按论文方法完成配对、消歧与抽样。

## 一、统一字段（13 列）

| 字段 | 含义 |
|---|---|
| corpus_id | 文本唯一编号（来源前缀_编号） |
| culture | 文化侧：`zh`（中文/中国侧）/ `west`（英文/西方侧） |
| genre | 体裁：`news` / `review` / `political_report` |
| source | 平台（douban / letterboxd / imdb_acl / imdb_genome / rmrb / finenews / gov.cn / govuk / presidency_ucsb / ec_europa 等） |
| channel | 频道/媒体名（如 人民日报、Independent、影片中文名、US State of the Union） |
| item | 归属条目（新闻源名 / 影片名 / 报告标题） |
| date | 日期（YYYY-MM-DD，部分缺失） |
| year | 年份 |
| rating | 评分（1–5 或 1–10，新闻无） |
| sentiment | 情感标签（aclImdb 有 pos/neg，其余空） |
| title | 标题 |
| text | 正文（已按各源清洗） |
| meta | JSON 附加信息（IP 属地、URL、字数、国家代码等） |

配对键：影评集为 `matched_film`（同片键），新闻集为 `match_month`（月份键），政治集以 `year` 配对。

## 二、文件清单

### 旗舰配对集（正文主分析）

| 文件 | 规模 | 说明 |
|---|---|---|
| reviews_samefilms_v3.parquet | 46,274 条（3,134 部影片，两侧各 23,137） | **主配对集**：豆瓣(2019)×Letterboxd(2025)同片影评，片名×年份（±1）双重消歧（剔除重拍片/同名剧集） |
| news_aligned_month.parquet | 27,000 篇（9 个月×双侧各 1,500） | **新闻主配对集**：人民日报系(2026-01~09)×FineNews 西方媒体(2025)，相邻年份同月份+议题分层 |
| reports_political.parquet | 122 份 | **政治话语层**：中国《政府工作报告》37 + 英国国王/女王演讲 34 + 美国 SOTU 37 + 欧盟 SOTEU 14（1990–2026，按年配对） |
| reviews_chinesefilms_matched.parquet | 9,744 条（618 部华语片） | 华语片同片配对（豆瓣×Letterboxd，内容归属调节分析） |
| reviews_douban_imdb_matched.parquet | 21,048 条（677 部） | 豆瓣(≤2011)×IMDB(2011)，aclImdb tt 编号直连，年代对齐 |
| reviews_samefilms_recent_v3.parquet | 2,492 条（188 部） | 同时代复核集：豆瓣 2020–2026 现抓 × Letterboxd 2025 |
| reviews_douban_genome_matched.parquet | 182,998 条（3,561 部） | 深度历史层：豆瓣(≤2012)×Tag Genome IMDB 评论，tt 直连 |

### 对照与稳健性集

| 文件 | 规模 | 说明 |
|---|---|---|
| news_general.parquet | 40,000 篇 | 历史新闻（THUCNews 2005–2011 × XSum/BBC 2010–2017） |
| news_recent_2026.parquet | 8,666 篇 | 近期窗口对照（中文 2026-08~09 × 西方 2025-08~10） |
| reviews_general.parquet | 20,000 条 | 历史影评（豆瓣 × IMDB 随机对照） |
| reviews_westernfilms.parquet | 20,000 条 | 豆瓣西片 × IMDB 对照 |
| reviews_samefilms_recent.parquet / _films.csv | — | 同片配对集配套的影片对照表（matched_film、douban_name、movie_id、db_year、lb_year、n_zh、n_west） |

### 效标基准（论文另用，不属本目录）

- WVS 第七波（2017–2022）跨国微观数据：中国 3,036 / 美国 2,596 / 英国 2,609 样本（Haerpfer *et al.*, 2022）。

## 三、关键处理（可复现）

- **片名×年份双重消歧**：豆瓣侧经电影信息维表以拉丁原名消歧，Letterboxd 侧按（片名×年份）分组，年度冲突影片全部剔除（同名重拍片/剧集不再错配）。
- **tt 直连**：aclImdb 与 Tag Genome 的评论带 IMDB 影片 tt 编号，与豆瓣电影维表的 IMDb 列精确直连，彻底摆脱片名匹配歧义。
- **政治长文档**：均值 2.4 万字符，测量时用"分块均值嵌入"（按段落切块、逐块嵌入、按块长加权均值），避免截断损失；新闻与影评语料统一截取前 2,000 字符。
- 数据说明文档另有 `data_corpus/docs/语料说明.md`（更详尽的来源与清洗日志，如需可向作者索取）。

## 四、许可与引用

数据用于学术研究复现。豆瓣/Letterboxd/IMDB 评论为用户生成内容（UGC），请遵守各平台条款；Tag Genome 为 CC BY-NC 3.0（须引用 Kotkov *et al.*, 2021 与 Vig *et al.*, 2012）；aclImdb 引自 Maas *et al.*, 2011；政治文本为各国政府公开文件。引用本文数据请同时引用论文。
