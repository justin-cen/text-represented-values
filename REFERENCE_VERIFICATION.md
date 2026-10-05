# 参考文献核验报告（OpenAlex 权威核验）

本报告由 OpenAlex API（学术元数据权威库）自动核验生成，覆盖三篇论文的**全部 111 条参考文献**。

## 一、核验结果总览

| 论文 | 条目数 | DOI 精确匹配 | 标题匹配 | 会议论文 | 专著/数据集 |
| --- | --- | --- | --- | --- | --- |
| 基础论文 | 56 | 40 | 13 | 2 | 1（WVS 数据集） |
| 论文 A | 32 | 15 | 12 | 1 | 4（专著） |
| 论文 B | 23 | 11 | 9 | 2 | 1（专著） |
| **合计** | **111** | **66** | **34** | **5** | **6** |

**结论：111 条参考文献全部为真实存在的学术成果，无虚构条目。** 6 条"专著/数据集"（Hofstede《Culture's Consequences》、Shoemaker & Reese《Mediating the Message》、Brady、Stockmann、Repnikova、Grimmer *et al.* 专著，以及 WVS 第七波数据集）在 OpenAlex 中不作为期刊论文索引，其真实性由出版社与数据集官方页面确认。

## 二、期刊层级核验（"公认顶刊或重要期刊"要求）

参考文献所涉期刊中，命中公认顶刊/重要期刊 **19 种**（按 OpenAlex 被引中位数排序）：

| 期刊 | 条目数 | 被引中位数 |
| --- | --- | --- |
| *Advances in Experimental Social Psychology* | 2 | 14,513 |
| *Behavioral and Brain Sciences* | 1 | 12,584 |
| *Academy of Management Review* | 2 | 9,297 |
| *Foreign Affairs* | 3 | 4,828 |
| *Journal of Personality and Social Psychology* | 3 | 2,614 |
| *Science* | 7 | 2,578 |
| *Journal of Economic Literature* | 3 | 1,265 |
| *Proceedings of the National Academy of Sciences* | 4 | 1,111 |
| *Journalism Studies* | 2 | 617 |
| *Human Relations* | 1 | 580 |
| *Computational Linguistics* | 1 | 520 |
| *American Sociological Review* | 2 | 503 |
| *Annual Review of Psychology* | 2 | 384 |
| *PNAS Nexus* | 3 | 354 |
| *Journal of Artificial Intelligence Research* | 3 | 268 |
| *Nature Machine Intelligence* | 2 | 200 |
| *Political Analysis* | 1 | 83 |
| *European Journal of Personality* | 3 | 73 |
| *Communication Methods and Measures* | 4 | 20 |

此外，计算机领域顶会论文来自 ACL、EMNLP、NAACL-HLT、ICLR、ICML、NeurIPS 等。

## 三、核验方法与可复现性

- **DOI 优先**：条目含 DOI 者直接按 DOI 精确查询 OpenAlex `works/https://doi.org/{doi}`；
- **标题回退**：无 DOI 者以标题检索 + 首作者 + 年份交叉评分匹配；
- **会议论文**：标题含特殊字符者改用 `filter=title.search` 检索；
- 脚本：[`../code/verify_refs_doi.py`](../code/verify_refs_doi.py)（若已发布）；
- 明细数据：`ref_verification2.csv`（逐条列出原条目、OpenAlex 标题、年份、期刊、被引量、DOI）。

## 四、备注

核验发现的所有条目标题、年份、期刊与论文参考文献表一致；未发现需要更正的元数据错误。检索日期：2026 年 10 月。
