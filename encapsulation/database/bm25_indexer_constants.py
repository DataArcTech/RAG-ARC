EXCLUDED_METADATA_FIELDS = {
    "score",  # Relevance scores should not be indexed as searchable fields
    "_score",  # Alternative score field name
    "rank",  # Ranking information
    "_rank",  # Alternative rank field name
    "distance",  # Distance/similarity measures
    "_distance",  # Alternative distance field name
    "similarity",  # Similarity scores
    "_similarity",  # Alternative similarity field name
}

