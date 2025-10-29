"""
HippoRAG2-specific extraction prompts optimized for minimal token usage with TSV format
Based on HippoRAG2's NER and Triple Extraction approach
All prompts output entity types in TSV format: entity\ttype
"""

# HippoRAG2 NER Prompt - Always outputs entity types
# If entity_types is specified: only extract those types
# If entity_types is None: LLM determines types automatically

HIPPORAG2_NER_SYSTEM = """Your task is to extract named entities from the given paragraph.
Respond with TSV (Tab-Separated Values) format: entity\ttype
Determine the entity type yourself (e.g., PERSON, ORGANIZATION, LOCATION, DATE, MONEY, etc.).
Minimize token usage by using concise output.
"""

HIPPORAG2_NER_SYSTEM_WITH_TYPES = """Your task is to extract named entities of specified types from the given paragraph.
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




