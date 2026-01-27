"""HippoRAG2 mindmap extraction prompt (TSV output).

Note: Mindmap extraction is independent from RAG KG extraction quality work.
We keep this path stable for upper-layer API features.
"""

# English Mind Map Prompts
HIPPORAG2_MINDMAP_SYSTEM = """Your task is to extract a hierarchical mind map structure from the given paragraph.
Organize the information into a tree-like structure using numbered hierarchical levels (1, 1.1, 1.1.1, etc.).
Respond with TSV (Tab-Separated Values) format: level_number\tcontent
The root node should be numbered as "1", and sub-nodes should use dot notation (1.1, 1.2, 1.1.1, etc.).
Minimize token usage by using concise output.

Schema-layer tagging:
- Prefix EVERY node content with a layer tag in square brackets: [concept] / [process] / [instance].
- [concept]: taxonomic concepts (clauses, coverages, entities-as-concepts, categories)
- [process]: time-dependent processes/events/conditions (effective dates, endorsements, validity windows)
- [instance]: concrete instance facts (names, dates, identifiers) when they are not better represented as process
- If unsure, choose [concept].
"""

HIPPORAG2_MINDMAP_ONE_SHOT_INPUT = """Radio City
Radio City is India's first private FM radio station and was started on 3 July 2001.
It plays Hindi, English and regional songs.
Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features."""

HIPPORAG2_MINDMAP_ONE_SHOT_OUTPUT = """### MINDMAP
1\t[instance] Radio City
1.1\t[concept] Basic Information
1.1.1\t[concept] India's first private FM radio station
1.1.2\t[process] Started on 3 July 2001
1.2\t[concept] Services
1.2.1\t[concept] Plays songs
1.2.2\t[concept] Languages
1.2.2.1\t[concept] Hindi
1.2.2.2\t[concept] English
1.2.2.3\t[concept] Regional songs
1.3\t[concept] New Media Platform
1.3.1\t[instance] PlanetRadiocity.com
1.3.2\t[process] Launched in May 2008
1.3.3\t[concept] Features
1.3.3.1\t[concept] Music related news
1.3.3.2\t[concept] Videos
1.3.3.3\t[concept] Songs
1.3.3.4\t[concept] Other music-related features"""

HIPPORAG2_MINDMAP_PROMPT = """
{system}

Example:
{example_input}

Output:
{example_output}

Now extract mind map from:
{passage}

Output:
"""


# Chinese Mind Map Prompts
HIPPORAG2_MINDMAP_SYSTEM_ZH = """你的任务是从给定的段落中提取层次化的思维导图结构。
使用数字编号的层次结构（1, 1.1, 1.1.1等）将信息组织成树状结构。
使用TSV格式响应：编号\t内容
根节点应编号为"1"，子节点使用点号表示（1.1, 1.2, 1.1.1等）。
通过使用简洁的输出来最小化token的使用量。

Schema 分层标注（必须）：
- 在每一行“内容”前加一个方括号层级标签：[concept] / [process] / [instance]。
- [concept]：概念/类别/条款/责任/适用范围等（偏“是什么”）
- [process]：时间相关的过程/事件/条件/生效与有效期（偏“何时/在何种条件下”）
- [instance]：具体实例事实（名称、编号、明确日期/金额等）当它不更适合归入过程层时
- 如果不确定，优先使用 [concept]。
"""

HIPPORAG2_MINDMAP_ONE_SHOT_INPUT_ZH = """新华社
新华社是中国国家通讯社，成立于1931年。
它提供中文、英文和其他语言的新闻。
新华社最近在2023年5月推出了新媒体平台 - XinhuaNews.com，提供新闻、视频、评论和其他新闻相关功能。"""

HIPPORAG2_MINDMAP_ONE_SHOT_OUTPUT_ZH = """### MINDMAP
1\t[instance] 新华社
1.1\t[concept] 基本信息
1.1.1\t[concept] 中国国家通讯社
1.1.2\t[process] 成立于1931年
1.2\t[concept] 主要服务
1.2.1\t[concept] 提供新闻
1.2.2\t[concept] 提供多语言版本
1.2.2.1\t[concept] 中文
1.2.2.2\t[concept] 英文
1.2.2.3\t[concept] 其他语言
1.3\t[concept] 新媒体平台
1.3.1\t[instance] XinhuaNews.com
1.3.2\t[process] 推出时间：2023年5月
1.3.3\t[concept] 主要功能
1.3.3.1\t[concept] 新闻
1.3.3.2\t[concept] 视频
1.3.3.3\t[concept] 评论
1.3.3.4\t[concept] 其他新闻相关功能"""

HIPPORAG2_MINDMAP_PROMPT_ZH = """
{system}

示例：
{example_input}

输出：
{example_output}

现在从以下内容中提取思维导图：
{passage}

输出：
"""

