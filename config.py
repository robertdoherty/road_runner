# Global configuration for labeling/enrichment behavior


########################### Concurrency settings ###########################
# Default concurrency for solution labeler LLM batch execution
DEFAULT_SOLUTION_MAX_CONCURRENCY = 10

# Default concurrency for break labeler batch execution
DEFAULT_BREAK_MAX_CONCURRENCY = 10   

# Default concurrency for diagnostic agent batch execution
DEFAULT_DIAGNOSTIC_MAX_CONCURRENCY = 10

########################### Confidence settings ###########################
# Minimum confidence required for accepting comment-derived fields
COMMENT_ENRICHMENT_MIN_CONFIDENCE = 0.6


########################### Paths ###########################
# Path to golden diagnostic chart
GOLDEN_DIAGNOSTIC_CHART_PATH = "data_labeler/rule_labeler/meta/golden_set_v3.json"

# Path to diagnostics ontology (labels)
DIAGNOSTICS_ONTOLOGY_PATH = "data_labeler/rule_labeler/meta/diagnostics_v1.json"

## Other settings

