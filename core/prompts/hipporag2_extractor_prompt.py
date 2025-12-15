
HIPPORAG2_NER_SYSTEM = """Your task is to extract disambiguated named entities from the given paragraph.
Resolve pronouns and ambiguous mentions (e.g., "this book" → actual book title) based on context.Exclude vague or unresolved entities if their referents cannot be determined.
Respond with TSV (Tab-Separated Values) format: entity\ttype
Determine the entity type yourself (e.g., PERSON, ORGANIZATION, LOCATION, DATE, MONEY, etc.).
Minimize token usage by using concise output.
"""

HIPPORAG2_NER_SYSTEM_WITH_TYPES = """Your task is to extract disambiguated named entities of specified types from the given paragraph.
Resolve pronouns and ambiguous mentions (e.g., "this book" → actual book title) based on context.Exclude vague or unresolved entities if their referents cannot be determined.
Respond with TSV (Tab-Separated Values) format: entity\ttype
Only extract entities that match the specified types.
Minimize token usage by using concise output.
"""

# Example for auto-determined entity types
HIPPORAG2_NER_ONE_SHOT_INPUT = """Radio City
Radio City is India's first private FM radio station and was started on 3 July 2001.
It plays Hindi, English and regional songs.
Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features."""

HIPPORAG2_NER_ONE_SHOT_OUTPUT = """### ENTITIES
Radio City\tORGANIZATION
India\tLOCATION
3 July 2001\tDATE
Hindi\tLANGUAGE
English\tLANGUAGE
May 2008\tDATE
PlanetRadiocity.com\tORGANIZATION"""

# Example for specified entity types
HIPPORAG2_NER_ONE_SHOT_INPUT_WITH_TYPES = """Radio City
Radio City is India's first private FM radio station and was started on 3 July 2001.
It plays Hindi, English and regional songs.
Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features."""

HIPPORAG2_NER_ONE_SHOT_OUTPUT_WITH_TYPES = """### ENTITIES
Radio City\tORGANIZATION
PlanetRadiocity.com\tORGANIZATION"""

# Prompt template (used when entity_types is None - LLM auto-determines types)
HIPPORAG2_NER_PROMPT = """
{system}

Example:
{example_input}

Output:
{example_output}

Now extract entities from:
{passage}

Output:
"""

# Prompt template (used when entity_types is specified)
HIPPORAG2_NER_PROMPT_WITH_TYPES = """
{system}

Entity types to extract: {entity_types}

Example:
{example_input}

Output:
{example_output}

Now extract entities from:
{passage}

Output:
"""


# HippoRAG2 Triple Extraction Prompt (TSV format for minimal tokens)
HIPPORAG2_TRIPLE_SYSTEM = """Your task is to construct an RDF graph from the given passage and named entities.
Respond with TSV (Tab-Separated Values) format triples to minimize token usage.

Requirements:
- Only use entities from the provided named entity list as subjects or objects
- Clearly resolve pronouns and referential expressions (e.g., "the book", "he", "it", "this") to their specific names from the named entity list
- If a pronoun or reference cannot be matched to an entity in the list, skip that triple
- Use tab-separated format: subject\tpredicate\tobject
"""

HIPPORAG2_TRIPLE_ONE_SHOT_INPUT = """Paragraph:
Radio City
Radio City is India's first private FM radio station and was started on 3 July 2001.
It plays Hindi, English and regional songs.
Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features.

Named Entities:
Radio City\tORGANIZATION
India\tLOCATION
3 July 2001\tDATE
Hindi\tLANGUAGE
English\tLANGUAGE
May 2008\tDATE
PlanetRadiocity.com\tORGANIZATION"""

HIPPORAG2_TRIPLE_ONE_SHOT_OUTPUT = """### TRIPLES
Radio City\tlocated in\tIndia
Radio City\tis\tprivate FM radio station
Radio City\tstarted on\t3 July 2001
Radio City\tplays songs in\tHindi
Radio City\tplays songs in\tEnglish
Radio City\tforayed into\tNew Media
Radio City\tlaunched\tPlanetRadiocity.com
PlanetRadiocity.com\tlaunched in\tMay 2008
PlanetRadiocity.com\tis\tmusic portal
PlanetRadiocity.com\toffers\tnews
PlanetRadiocity.com\toffers\tvideos
PlanetRadiocity.com\toffers\tsongs"""

HIPPORAG2_TRIPLE_PROMPT = """
{system}

Example:
{example_input}

Output:
{example_output}

Now extract triples from:
Paragraph:
{passage}

Named Entities:
{entities}

Output:
"""


# ============================================================================
# Chinese (中文) Prompts
# ============================================================================

HIPPORAG2_NER_SYSTEM_ZH = """你的任务是从给定的段落中提取消歧后的命名实体。
根据上下文确定实体的确切指代，例如“本书”应该替换成书名，“他”应该替换成具体人物。若无法确定明确实体，则忽略该实体，不输出模糊实体。
使用TSV格式响应：实体\t类型
自己确定实体类型（例如，人物、组织、地点、日期、金额等）。
通过使用简洁的输出来最小化token的使用量。
"""

HIPPORAG2_NER_SYSTEM_WITH_TYPES_ZH = """你的任务是从给定的段落中提取指定类型的消歧后的命名实体。
根据上下文确定实体的确切指代，例如“本书”应该替换成书名，“他”应该替换成具体人物。若无法确定明确实体，则忽略该实体，不输出模糊实体。
使用TSV格式响应：实体\t类型
仅提取与指定类型匹配的实体。
通过使用简洁的输出来最小化token的使用量。
"""

HIPPORAG2_NER_ONE_SHOT_INPUT_ZH = """新华社
新华社是中国国家通讯社，成立于1931年。
它提供中文、英文和其他语言的新闻。
新华社最近在2023年5月推出了新媒体平台 - XinhuaNews.com，提供新闻、视频、评论和其他新闻相关功能。"""

HIPPORAG2_NER_ONE_SHOT_OUTPUT_ZH = """### ENTITIES
新华社\t组织
中国\t地点
1931年\t日期
中文\t语言
英文\t语言
2023年5月\t日期
XinhuaNews.com\t组织"""

# Example for specified entity types (Chinese)
HIPPORAG2_NER_ONE_SHOT_INPUT_WITH_TYPES_ZH = """新华社
新华社是中国国家通讯社，成立于1931年。
它提供中文、英文和其他语言的新闻。
新华社最近在2023年5月推出了新媒体平台 - XinhuaNews.com，提供新闻、视频、评论和其他新闻相关功能。"""

HIPPORAG2_NER_ONE_SHOT_OUTPUT_WITH_TYPES_ZH = """### ENTITIES
新华社\t组织
XinhuaNews.com\t组织"""

# Prompt template (Chinese - used when entity_types is None)
HIPPORAG2_NER_PROMPT_ZH = """
{system}

示例：
{example_input}

输出：
{example_output}

现在从以下内容中提取实体：
{passage}

输出：
"""

# Prompt template (Chinese - used when entity_types is specified)
HIPPORAG2_NER_PROMPT_WITH_TYPES_ZH = """
{system}

要提取的实体类型：{entity_types}

示例：
{example_input}

输出：
{example_output}

现在从以下内容中提取实体：
{passage}

输出：
"""


# HippoRAG2 Triple Extraction Prompt - Chinese Version
HIPPORAG2_TRIPLE_SYSTEM_ZH = """你的任务是从给定的段落和命名实体构建RDF图。
使用TSV格式三元组来最小化token的使用量。

要求：
- 只能使用提供的命名实体列表中的实体作为主语或宾语
- 如果段落中有代词或指代词(如"本书"、"他"、"它"等),必须将其替换为命名实体列表中对应的具体实体
- 如果无法从命名实体列表中找到对应的实体,则跳过该三元组
- 使用制表符分隔格式：主语\t谓语\t宾语
"""

HIPPORAG2_TRIPLE_ONE_SHOT_INPUT_ZH = """段落：
新华社
新华社是中国国家通讯社，成立于1931年。
它提供中文、英文和其他语言的新闻。
新华社最近在2023年5月推出了新媒体平台 - XinhuaNews.com，提供新闻、视频、评论和其他新闻相关功能。

命名实体：
新华社\t组织
中国\t地点
1931年\t日期
中文\t语言
英文\t语言
2023年5月\t日期
XinhuaNews.com\t组织"""

HIPPORAG2_TRIPLE_ONE_SHOT_OUTPUT_ZH = """### TRIPLES
新华社\t位于\t中国
新华社\t是\t国家通讯社
新华社\t成立于\t1931年
新华社\t提供新闻\t中文
新华社\t提供新闻\t英文
新华社\t推出\t新媒体平台
新华社\t推出\tXinhuaNews.com
XinhuaNews.com\t推出于\t2023年5月
XinhuaNews.com\t是\t新媒体平台
XinhuaNews.com\t提供\t新闻
XinhuaNews.com\t提供\t视频
XinhuaNews.com\t提供\t评论"""

HIPPORAG2_TRIPLE_PROMPT_ZH = """
{system}

示例：
{example_input}

输出：
{example_output}

现在从以下内容中提取三元组：
段落：
{passage}

命名实体：
{entities}

输出：
"""


# ============================================================================
# Mind Map Extraction Prompts
# ============================================================================

# English Mind Map Prompts
HIPPORAG2_MINDMAP_SYSTEM = """Your task is to extract a hierarchical mind map structure from the given paragraph.
Organize the information into a tree-like structure using numbered hierarchical levels (1, 1.1, 1.1.1, etc.).
Respond with TSV (Tab-Separated Values) format: level_number\tcontent
The root node should be numbered as "1", and sub-nodes should use dot notation (1.1, 1.2, 1.1.1, etc.).
Minimize token usage by using concise output.
"""

HIPPORAG2_MINDMAP_ONE_SHOT_INPUT = """Radio City
Radio City is India's first private FM radio station and was started on 3 July 2001.
It plays Hindi, English and regional songs.
Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features."""

HIPPORAG2_MINDMAP_ONE_SHOT_OUTPUT = """### MINDMAP
1	Radio City
1.1	Basic Information
1.1.1	India's first private FM radio station
1.1.2	Started on 3 July 2001
1.2	Services
1.2.1	Plays songs
1.2.2	Languages
1.2.2.1	Hindi
1.2.2.2	English
1.2.2.3	Regional songs
1.3	New Media Platform
1.3.1	PlanetRadiocity.com
1.3.2	Launched in May 2008
1.3.3	Features
1.3.3.1	Music related news
1.3.3.2	Videos
1.3.3.3	Songs
1.3.3.4	Other music-related features"""

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
"""

HIPPORAG2_MINDMAP_ONE_SHOT_INPUT_ZH = """新华社
新华社是中国国家通讯社，成立于1931年。
它提供中文、英文和其他语言的新闻。
新华社最近在2023年5月推出了新媒体平台 - XinhuaNews.com，提供新闻、视频、评论和其他新闻相关功能。"""

HIPPORAG2_MINDMAP_ONE_SHOT_OUTPUT_ZH = """### MINDMAP
1	新华社
1.1	基本信息
1.1.1	中国国家通讯社
1.1.2	成立于1931年
1.2	主要服务
1.2.1	提供新闻
1.2.2	提供多语言版本
1.2.2.1	中文
1.2.2.2	英文
1.2.2.3	其他语言
1.3	新媒体平台
1.3.1	XinhuaNews.com
1.3.2	推出时间：2023年5月
1.3.3	主要功能
1.3.3.1	新闻
1.3.3.2	视频
1.3.3.3	评论
1.3.3.4	其他新闻相关功能"""

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