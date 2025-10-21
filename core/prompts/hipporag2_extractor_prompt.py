"""
HippoRAG2-specific extraction prompts optimized for minimal token usage with TSV format
Based on HippoRAG2's NER and Triple Extraction approach
"""

# HippoRAG2 NER Prompt (TSV format for minimal tokens)
HIPPORAG2_NER_SYSTEM = """Your task is to extract named entities from the given paragraph.
Respond with a TSV (Tab-Separated Values) list of entities to minimize token usage.
"""

HIPPORAG2_NER_ONE_SHOT_INPUT = """Radio City
Radio City is India's first private FM radio station and was started on 3 July 2001.
It plays Hindi, English and regional songs.
Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features."""

HIPPORAG2_NER_ONE_SHOT_OUTPUT = """### ENTITIES
Radio City
India
3 July 2001
Hindi
English
May 2008
PlanetRadiocity.com"""

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


# HippoRAG2 Triple Extraction Prompt (TSV format for minimal tokens)
HIPPORAG2_TRIPLE_SYSTEM = """Your task is to construct an RDF graph from the given passage and named entities.
Respond with TSV (Tab-Separated Values) format triples to minimize token usage.

Requirements:
- Each triple should contain at least one, but preferably two, of the named entities
- Clearly resolve pronouns to their specific names
- Use tab-separated format: subject\tpredicate\tobject
"""

HIPPORAG2_TRIPLE_ONE_SHOT_INPUT = """Paragraph:
Radio City
Radio City is India's first private FM radio station and was started on 3 July 2001.
It plays Hindi, English and regional songs.
Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features.

Named Entities:
Radio City
India
3 July 2001
Hindi
English
May 2008
PlanetRadiocity.com"""

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


# Combined HippoRAG2 OpenIE Prompt (NER + Triple Extraction in one call)
HIPPORAG2_OPENIE_SYSTEM = """Your task is to extract named entities and construct an RDF graph from the given paragraph.
Respond with TSV (Tab-Separated Values) format to minimize token usage.

Requirements:
1. First extract all named entities
2. Then construct triples using these entities
3. Each triple should contain at least one, but preferably two, of the named entities
4. Clearly resolve pronouns to their specific names
"""

HIPPORAG2_OPENIE_ONE_SHOT_INPUT = """Radio City
Radio City is India's first private FM radio station and was started on 3 July 2001.
It plays Hindi, English and regional songs.
Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features."""

HIPPORAG2_OPENIE_ONE_SHOT_OUTPUT = """### ENTITIES
Radio City
India
3 July 2001
Hindi
English
May 2008
PlanetRadiocity.com

### TRIPLES
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

HIPPORAG2_OPENIE_PROMPT = """
{system}

Example:
{example_input}

Output:
{example_output}

Now extract from:
{passage}

Output:
"""


# Chinese version for bilingual support
HIPPORAG2_OPENIE_SYSTEM_ZH = """你的任务是从给定段落中提取命名实体并构建RDF图。
使用TSV（制表符分隔值）格式响应以最小化token使用。

要求：
1. 首先提取所有命名实体
2. 然后使用这些实体构建三元组
3. 每个三元组应包含至少一个，最好两个命名实体
4. 清楚地将代词解析为具体名称
"""

HIPPORAG2_OPENIE_ONE_SHOT_INPUT_ZH = """北京大学
北京大学创办于1898年，初名京师大学堂，是中国第一所国立综合性大学。
1912年改为现名。作为新文化运动的中心和五四运动的策源地，北京大学为民族的振兴和解放、国家的建设和发展、社会的文明和进步做出了不可替代的贡献。"""

HIPPORAG2_OPENIE_ONE_SHOT_OUTPUT_ZH = """### ENTITIES
北京大学
1898年
京师大学堂
中国
1912年
新文化运动
五四运动

### TRIPLES
北京大学\t创办于\t1898年
北京大学\t初名\t京师大学堂
京师大学堂\t是\t中国第一所国立综合性大学
北京大学\t改名于\t1912年
北京大学\t是中心\t新文化运动
北京大学\t是策源地\t五四运动
北京大学\t做出贡献于\t民族振兴
北京大学\t做出贡献于\t国家建设
北京大学\t做出贡献于\t社会进步"""

